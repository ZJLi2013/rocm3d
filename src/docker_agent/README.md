# docker_agent



## Overview 

完整流程（当前目标形态）：

1. 执行 `docker_agent` install 阶段（`--install-script`）。
2. 若 install 成功：
   - 提交镜像（默认 commit，除非显式 `--no-commit-image`）；
   - 进入 run 阶段（`--run-script`，或后续生成的 sample 脚本）。
3. 若 install 失败：
   - 运行 LLM 分析得到 `llm_analysis_plan`（`root_cause` + `execution_plan`）；
   - review 后将 `patches` 应用到 `install_script`；
   - rerun install（默认最多 3 次），直到通过或进入 `need_human`

4. TODO：自动生成/收敛 run sample 脚本（`sample.py` 或等价 run script），形成 install->run 的闭环自动化。

**暂时先考虑 IDE 生成 sample.py**




## 入口

- **命令行入口**：`python -m docker_agent`（会执行包内的 `__main__.py`）。
- 从代码调用：`from docker_agent import DockerAgent`，使用 `DockerAgent().build_image(BuildRequest(...))`。

## 代码结构（文件职责）

| 文件 | 职责 |
|------|------|
| `__main__.py` | **CLI 入口**：`--repo_url` / `--base-image` / `--install-script`（或 `--install-cmd`+`--run-cmd`）/ `--run-script` / `--run-timeout` / `--discard-workspace`，输出 JSON |
| `agent.py` | 顶层入口编排：`DockerAgent`、`BuildRequest`/`BuildResult`，负责 workspace/repo 准备并调用 orchestrator |
| `orchestrator.py` | 执行编排层：install/run 执行、失败分类、LLM 分析调用、auto-patch 重试控制 |
| `llm_analyzer.py` | LLM 分析层：对 `llm_log_analyzer.py` 的封装，提供统一 analyzer 接口 |
| `patch_engine.py` | patch 引擎层：对 `patch_from_llm_analysis.py` 的封装，提供统一 patch apply 接口 |
| `container_runner.py` | 容器执行：`run_container(..., script_path= 或 install_cmd+run_cmd)` → `RunResult`（docker-py） |
| `workspace.py` | 临时工作目录：`WorkspaceManager`、`Workspace` |
| `repo_manager.py` | Git 操作：`RepoManager.clone()` |
| `error_classifier.py` | 失败日志提取与规则分类（`failure_type` / `retryable` / `signals`） |
| `llm_log_analyzer.py` | 读取 install/run 失败上下文，调用 LLM 生成结构化修复计划（dry-run，不直接改脚本） |

## `llm_log_analyzer.py` 说明

### 主要能力

- 输入：阶段（`install`/`run`）、repo/base image、错误分类结果、stdout/stderr、脚本文本。
- 行为：抽取关键错误片段，构造 JSON prompt，请求 LLM 返回标准 JSON 计划。
- 输出：`LLMAnalysisPlan`（`root_cause` + `execution_plan`）。
- 失败兜底：若 API key 缺失、网关不可达、返回非 JSON，会返回 `execution_plan.action=need_human`。

### 关键函数

- `analyze_failure_with_llm(...)`：主入口，负责 prompt 组装、重试、错误处理。
- `parse_llm_analysis(raw)`：解析 LLM 文本为结构化计划；解析失败时返回 `action=need_human` 的兜底结果。
- `_extract_json_block(raw)`：兼容 ```json ... ``` 与裸 JSON。
- `_looks_like_gateway_connectivity_issue(...)`：识别连接类错误并补充诊断提示。

### Prompt 资产文件

当前 `llm_log_analyzer.py` 会从 `src/docker_agent/prompts/` 读取以下文件：

- `analyzer_system.txt`：system prompt 文本
- `analyzer_constraints.txt`：约束列表（按行读取；空行和 `#` 注释行会忽略）
- `analyzer_output_schema.json`：`output_schema` 模板
- `analyzer_fewshot.md`：few-shot 纯文本片段（可选）

回退策略（兼容性）：

- 文件缺失或解析失败时，自动回退到代码内置默认值；
- 无 few-shot 文件或为空时，`fewshot_examples` 会是空字符串。


### 经验沉淀 `.md` 注入（已支持最小版）

目标：把多 repo 复盘沉淀（错误模式、修复策略、反例）以可控方式注入 analyzer prompt。

当前行为：

1. `llm_log_analyzer.py` 会扫描 `docs/skills/*.md`。
2. 基于 `failure_type` 与 `repo_url` 做轻量匹配，提取 markdown 中的 bullet/编号条目。
3. 结果注入到 prompt payload 的 `experience_snippets` 字段（最多若干条）。
4. 若目录不存在或无可用片段，注入空列表，不影响主流程。

建议约定：

- 在 `docs/skills/` 中按主题拆分经验文件（如 `apt_package.md`、`rocm_common.md`）；
- 每条经验尽量保持“触发条件 + 操作建议”的短句格式，便于提取器复用；
- 避免将长段落/无关背景写成 bullet，减少 prompt 噪声。

实践建议：

- 优先注入“可执行修复模式 + 适用条件 + 禁用反例”；
- few-shot 仅保留高质量短样例，避免过多示例稀释约束；
- 每条经验尽量可验证（例如“Ubuntu 24.04 用 `libegl1` 而非 `libegl1-mesa`”）。

## 使用

**依赖**：`pip install -r requirements.txt`（含 `docker`）。

- **仅 clone**：不传 `--base-image` 或只传 `--base-image` 不传 run 相关参数。
- **简单 run**：`--install-cmd` + `--run-cmd`（单行 install + 单行 run，适合仅 `pip install -r requirements.txt` 的 repo）。
- **落地推荐**：`--install-script <路径>`（及可选 `--run-script`），脚本拷贝进 repo 或挂载后执行（conda + 多步 pip + 运行）。
- **自动闭环修复（install）**：`--auto-patch-on-fail --max-auto-patch-retries 3`（失败时自动 LLM 分析并按 action 进行 patch/retry）。


Repo 内脚本（conda + 多步安装 + 运行）：

```bash
PYTHONPATH=./src python -m docker_agent --repo_url <git_url> --base-image <image> \
  --install-script samples/manually_scripts/dust3r.sh \
  --auto-patch-on-fail --max-auto-patch-retries 3 \
  [--run-timeout 3600]
```

输出 JSON 示例（成功，含 run 结果）：

```json
{
  "status": "success",
  "repo_path": "/tmp/docker_agent_xxx/repo",
  "workspace_path": "/tmp/docker_agent_xxx",
  "message": null,
  "run_exit_code": 0,
  "run_stdout": "...",
  "run_stderr": "",
  "run_timed_out": false
}
```

## LLM 分析：输入 / 输出示例

### 1) 输入示例（docker_agent 结果 JSON）

`tools/run_llm_analysis_from_result.py` 会读取一个结果文件（例如 `samples/test_output/depth_anything_3.json`），其中 install 相关字段如下：

```json
{
  "status": "failed",
  "message": "install failed",
  "install_exit_code": 1,
  "install_stdout": "... ERROR: No matching distribution found for ninja",
  "install_stderr": "",
  "run_exit_code": null,
  "run_stdout": null,
  "run_stderr": null
}
```

运行命令示例（只分析 install）：

```bash
PYTHONPATH=./src python tools/run_llm_analysis_from_result.py \
  --input-json samples/test_output/depth_anything_3.json \
  --output-json samples/test_output/depth_anything_3_llm_analysis.json \
  --repo-url https://github.com/ByteDance-Seed/Depth-Anything-3 \
  --base-image rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0 \
  --stage install \
  --script-path samples/manually_scripts/depth_anything_3.sh
```

### 2) 输出示例（结构化修复计划）

输出文件（例如 `samples/test_output/depth_anything_3_llm_analysis.json`）：

```json
{
  "source_json": "samples/test_output/depth_anything_3_rerun.json",
  "stage": "install",
  "failure_type": "unknown",
  "retryable": false,
  "classifier_confidence": 0.2,
  "llm_analysis_plan": {
    "root_cause": {
      "evidence": [
        "ERROR: No matching distribution found for ninja"
      ],
      "why": "gsplat install uses a single index that cannot resolve transitive dependency ninja"
    },
    "execution_plan": {
      "action": "patch_script",
      "patches": [
        {
          "op": "replace_line",
          "target": "install_script",
          "match": "pip install gsplat --index-url=https://pypi.amd.com/simple",
          "content": "python -m pip install --extra-index-url https://pypi.amd.com/simple gsplat"
        }
      ]
    }
  }
}
```

### 3) 字段语义（极简结构）

- `root_cause`
  - `evidence`：来自日志的关键证据片段（字符串数组）。
  - `why`：对根因的简要解释（给人看的结论）。
- `execution_plan`
  - `action`：下一步动作（`patch_script` / `retry` / `need_human`）。
  - `patches`：仅当需要改脚本时给出 patch 列表。

### 4) `action` 消费建议

- `patch_script`：先人工 review `patches`，再应用并重试。
- `retry`：不改脚本，直接重试。
- `need_human`：停止自动修复，转人工处理。

### 5) `patches` 字段语义

每个 patch 的最小结构：

- `op`：编辑类型，当前支持 `replace_line` / `append_block` / `prepend_block`
- `target`：目标脚本，`install_script` 或 `run_script`
- `match`：定位锚点（用于替换时匹配原文本）
- `content`：要写入的新内容

注意：

- 当前 `llm_log_analyzer.py` 只负责“给计划”，不直接执行 patch。
- 执行侧应保证幂等和安全（例如 `match` 不命中时停止、记录审计日志）。

### 6) 如何扩展 `action` types（重点）

当前代码把 `action` 作为**白名单枚举**处理；新增类型时需要同步修改以下位置：

1. `src/docker_agent/llm_log_analyzer.py` 的类型定义
   - `ExecutionAction = Literal[...]` 加入新值（例如 `switch_base_image`）。
2. `parse_llm_analysis(...)` 的合法值校验
   - `if action not in {...}: action = "need_human"` 的集合要加新值。
3. `analyze_failure_with_llm(...)` 的 prompt schema
   - `output_schema.execution_plan.action` 文本枚举要加新值，保证模型知道可选项。
4. 文档与调用方分支逻辑
   - 本 README 的语义说明；
   - 任何消费 `action` 的执行器/编排器（避免新值被当未知值）。
5. 测试用例
   - 建议在 `tests/test_llm_log_analyzer.py` 增加“新 action 可解析/可降级”用例。

推荐扩展原则：

- 先定义“动作语义”再加类型（每个 action 必须对应明确后续动作）。
- 新类型优先少而稳，避免语义重叠（例如不要同时有 `retry` 和 `retry_now`）。
- 对未知 action 保持安全兜底（继续回退 `need_human`）。







## LLM 分析后 Patch（通用流程）

建议按这个最小闭环执行：

1. **Review 分析结果**
   - 确认 `root_cause.evidence` 与失败日志一致；
   - 确认 `execution_plan.action` 合理（`patch_script` / `retry` / `need_human`）。
2. **若 `action=patch_script`，先打最小 patch**
   - 仅应用 `patches` 中与当前错误直接相关的修改（避免一次改太多）。
   - 可用通用工具将 `llm_analysis.json` 应用到脚本：

```bash
PYTHONPATH=./src python tools/apply_patches_from_llm_analysis.py \
  --analysis-json samples/test_output/<case>_llm_analysis.json \
  --install-script-path samples/manually_scripts/<install_script>.sh \
  --output-json samples/test_output/<case>_patch_apply.json
```

   - 若有 run 阶段 patch，可补充 `--run-script-path <path>`。
   - 当前实现中，只有 `action=patch_script` 会触发显式 patch；`retry` 与 `need_human` 由上层编排决策（不自动改脚本）。
3. **重新执行 install 验证**
   - 在远端 GPU 节点运行 `docker_agent` install 阶段，生成新的结果 JSON。
4. **再次运行 LLM 分析**
   - 用新的结果 JSON 重新执行 `run_llm_analysis_from_result.py`。
5. **达成结束态**
   - install 成功：继续 run 阶段；
   - 超过重试上限仍失败，或 LLM 返回 `need_human`：结束并返回 `status=need_human`。

> 将 `llm_analysis.json` 回传本地做对比属于调试/复盘手段，不是主流程必需步骤。

后续扩展建议：

- 为 `retry` 增加可配置重试策略（重试次数、退避、网络类错误白名单）。
- 为 `need_human` 增加标准化输出（工单模板、阻塞原因分类、建议人工动作）。
- 若后续引入新 `action`（如 `switch_base_image`），优先在编排层新增分支，保持 patch 引擎职责单一。

## 输出文件与来源（重要）

为避免混淆，闭环相关输出可按“谁生成”理解：

### 1) `*_autoloop.json`（主流程结果）

- 典型文件：`samples/test_output/depth_anything_3_autoloop.json`
- 生成者：`python -m docker_agent ... -o <output>`（`__main__.py` + `agent.py`）
- 含义：一次完整编排执行的结果快照（install/run 状态、日志、分类、llm_analysis_plan、patch_apply_result 等）。
- 说明：这不是 LLM 原始输出，也不是 patch 模块单独输出，而是主流程聚合结果。

### 2) `*_llm_analysis.json`（LLM 分析结果）

- 典型文件：`samples/test_output/depth_anything_3_llm_analysis.json`
- 生成者：`tools/run_llm_analysis_from_result.py`（内部调用 `llm_log_analyzer.py`）
- 含义：针对某次失败结果做结构化分析，核心是：
  - `root_cause`
  - `execution_plan`（`action` + `patches`）
- 说明：这是给“决策/修复”使用的输入，不直接执行 patch。

### 3) `*_patch_apply.json`（显式 patch 执行结果）

- 典型文件：`samples/test_output/depth_anything_3_patch_apply.json`
- 生成者：`tools/apply_patches_from_llm_analysis.py`（内部调用 `patch_from_llm_analysis.py`）
- 含义：将 `execution_plan.patches` 应用到脚本后的执行记录（改了哪些文件、每条 patch 的状态、错误信息）。
- 说明：该文件反映“patch 执行器”行为，不代表 install/run 是否成功。

### 4) 推荐读取顺序

1. 先看 `*_autoloop.json`：判断主流程最终成功、失败还是 `need_human`。
2. 再看 `llm_analysis_plan`（在 autoloop 内或独立 `*_llm_analysis.json`）：理解根因与建议动作。
3. 若有 patch 过程，再看 `*_patch_apply.json`：确认 patch 是否真实落地、是否有命中/冲突问题。

## Unit Test（推荐）

运行核心单测：

```bash
PYTHONPATH=./src python -m unittest \
  tests/test_llm_log_analyzer.py \
  tests/test_patch_from_llm_analysis.py \
  tests/test_agent_autopatch_loop.py
```

覆盖范围：

- `test_llm_log_analyzer.py`：LLM 结构化输出解析与兜底行为
- `test_patch_from_llm_analysis.py`：`execution_plan.patches` 显式 patch 执行逻辑
- `test_agent_autopatch_loop.py`：主流程闭环（`patch_script` / `retry` / `need_human`）编排行为




## 参考

- 项目根 [README.md](../../README.md)、[plan.md](../../plan.md)。
