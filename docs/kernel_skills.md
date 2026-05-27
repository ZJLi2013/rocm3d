# Kernel Skills 系统梳理

本文整理 `github/ai_agents/` 本地已有的 NVIDIA kernel skills 生态，目标是系统理解这些项目的分层、职责边界和可借鉴方法。这里不讨论如何并入 `rocm3d`。

主要阅读对象：

- `kernel-design-agents/`
- `KernelWiki/`
- `ncu-report-skill/`
- `kernel-pilot/`

相关 upstream：

- [mit-han-lab/kernel-design-agents](https://github.com/mit-han-lab/kernel-design-agents)
- [mit-han-lab/KernelWiki](https://github.com/mit-han-lab/KernelWiki)
- [BBuf/kernel-pilot](https://github.com/BBuf/kernel-pilot)

## 1. 总体结构

现在 NVIDIA kernel skills 不是单个“优化 prompt”，而是一个分层系统：

| 层级 | 本地目录 | 作用 |
|---|---|---|
| 通用工作流 | `kernel-design-agents` | 定义性能敏感任务的证据循环：任务契约、计划、候选实现、验证、评估、晋级。 |
| 知识库 | `KernelWiki` | 面向 Blackwell/Hopper kernel 的结构化 prior-art 知识库。 |
| profiler 诊断 | `ncu-report-skill` | 基于 Nsight Compute 的 B200/SM100 kernel profiling、诊断和报告流程。 |
| 自动优化循环 | `kernel-pilot` | 把 Humanize/RLCR、KernelWiki、ncu-report-skill 串成 kernel tuning loop。 |

核心思想是拆分职责，而不是把所有内容写进一个超长 skill：

```text
任务契约
  -> 独立实验 workspace
  -> baseline / oracle / workload
  -> prior-art 查询
  -> candidate 实现
  -> 正确性验证
  -> benchmark
  -> 必要时 profiler 诊断
  -> ledger 记录
  -> 晋级 / 拒绝决策
```

## 2. Kernel Design Agents：最小证据循环

`kernel-design-agents` 是底层通用 workflow。它不绑定 CUDA、B200 或某个 benchmark，重点是让 agent 在性能任务里按证据推进。

它要求先定义 task contract：

- 目标是什么。
- 输入输出是什么。
- 正确性要求是什么。
- 实现约束是什么。
- validation command 是什么。
- evaluation command 是什么。
- 什么条件下候选实现可以 promotion。

推荐 workspace：

```text
task-workspace/
  docs/
    draft.md
    plan.md
  runs/
  outputs/
  profile/
  benchmark.csv
  candidates.jsonl
```

关键规则：

- 先写 `docs/draft.md`，再形成可执行计划。
- 每次只实现一个 candidate。
- 每次 meaningful change 后验证。
- 失败候选也要记录原因，不要从历史里消失。

它的价值是把 agent 从“直接写代码”拉回到“有契约、有证据、有晋级标准”的工程循环。

## 3. KernelWiki：结构化 prior-art 知识库

`KernelWiki` 是这套系统里最像“长期知识资产”的部分。它不是普通 markdown 集合，而是有 schema、索引、查询工具、confidence、source trace 的知识库。

范围：

- 主要面向 Blackwell / SM100 / B200。
- Hopper / SM90 内容只有在有 Blackwell relevance 时保留。
- 只关注 kernel，不覆盖 host-side serving、调度策略、分布式系统。
- 一等公民 DSL：CuTe DSL、CUDA C++、PTX、Triton。

目录分层：

| 层 | 目录 | 作用 |
|---|---|---|
| 原始来源 | `sources/` | PR、官方文档、博客、比赛报告。 |
| 综合知识 | `wiki/` | hardware、technique、kernel、pattern、language、migration 页面。 |
| 自动索引 | `queries/` | 按 problem、technique、hardware feature、repo、kernel type、language 生成索引。 |

常用查询方式：

```bash
python3 scripts/query.py "ping-pong attention" --limit 5
python3 scripts/query.py --tag nvfp4 --type kernel
python3 scripts/query.py --architecture B200 --type kernel
python3 scripts/get_page.py kernel-flash-attention-4 --follow-sources
python3 scripts/grep_wiki.py "tcgen05\\.fence" --only wiki
```

知识分类覆盖面很广：

- Hardware：`tcgen05`、TMEM、CLC、TMA、2-SM cooperative、NVFP4、mbarrier。
- Technique：warp specialization、persistent kernel、ping-pong scheduling、epilogue fusion、pipeline stages、swizzling、tile scheduling、register budgeting、cache policy、vectorized loads。
- Kernel case：FlashAttention-4、DeepGEMM、NVFP4 GEMM/GEMV、FP8 block-scale GEMM、fused MoE、grouped GEMM、FlashMLA、NSA、Gated Delta Net。
- Pattern：low SM utilization、memory-bound、register pressure、compute-bound、tail effect、pipeline stalls、MoE load imbalance。
- Migration：Hopper `wgmma` 到 Blackwell `tcgen05`，register accumulator 到 TMEM。

证据体系：

| Confidence | 含义 |
|---|---|
| `verified` | 同时有 official-doc 和 upstream-code 证据。 |
| `source-reported` | 来自权威来源，如论文、官方博客、主流 repo。 |
| `inferred` | 多来源综合推断，引用时要说明。 |
| `experimental` | 未文档化或版本敏感，需要强 caveat。 |

可复现等级：

| Level | 含义 |
|---|---|
| `concept` | 只有文字概念。 |
| `pseudocode` | 算法伪代码。 |
| `snippet` | 可编译片段。 |
| `runnable` | 自包含可运行示例。 |
| `benchmarked` | 有环境信息和性能数据。 |

最值得学习的习惯：任何影响实现的结论，都要能追到 `sources:`，再追到 PR、artifact、官方文档或 upstream source path。

## 4. ncu-report-skill：profiling 诊断层

`ncu-report-skill` 的黄金规则是：

```text
先 profile，再诊断，再制定计划；不要靠猜。
```

它面向 B200 / SM100 / Nsight Compute，但流程本身很通用。

标准流程：

1. 在 `profile/<run_name>/` 下创建独立 run 目录。
2. 明确要 profile 的 shape、dispatch path 和问题。
3. 尽量构建 standalone harness。
4. 收集两份 NCU profile：
   - `--set full`：总览和 PM sampling。
   - `--set source --section SourceCounters`：source-line stall attribution。
5. 用 `ncu_report` Python API 解析，不靠肉眼看 CLI。
6. 按六个维度诊断。
7. 对照 diagnosis playbook 找原因和 first-line fix。
8. 写 `REPORT.md`，按证据和预期收益排序下一步。

六个诊断维度：

| 维度 | 要回答的问题 |
|---|---|
| SM occupancy / launch geometry | grid 是否足够填满 GPU，occupancy 被什么限制。 |
| Thread-block balance / tail effect | CTA 或 SM 是否不均衡，是否有长尾。 |
| Stall reason / source hotspot | warp 在等什么，哪几行代码导致 stall。 |
| Tensor Core utilization | matmul-like work 是否真的用了 tensor core，利用率如何。 |
| SM utilization timeline | timeline 是平稳、低利用率、长尾还是 sawtooth pipeline bubble。 |
| Memory system | global/shared/local access 是否 coalesced、vectorized、复用良好。 |

常见 diagnosis pattern：

- Small grid / SM idle。
- 变长输入导致 tail effect。
- Uncoalesced global loads。
- Sparse writes。
- Long-scoreboard latency-bound。
- Compute-bound 但没用 tensor cores。
- Atomics contention。
- Shared-memory bank conflicts。

这个 skill 强调“具体数值是交付物”：不能只说 memory-bound，而要指出哪些 metric 支撑这个结论。

## 5. KernelPilot：kernel 优化闭环

`kernel-pilot` 是集成层，把 workflow、知识库、profiling 和 review loop 组合起来。

它围绕三个输入启动：

```text
K: kernel definition and semantics
R: correctness reference or oracle
W: workload distribution or focused benchmark case
```

如果 `K/R/W`、目标 GPU、baseline、comparison target 或硬约束缺失，就先问清楚，不直接优化。

KernelPilot 的关键约束：

- 只使用一个干净的 standalone optimization workspace。
- 大型 framework repo 默认只读，除非用户明确要求 in-place patch。
- 先创建 scaffold、harness placeholder、ledger、refined plan。
- workspace 必须是 git repo，并且有一个干净的 scaffold commit。
- 启动 Humanize RLCR，并要求 `--strict-success`。
- 确认 `.humanize/rlcr/<timestamp>/state.md` 存在后，才进入 candidate kernel 实现。

典型 workspace：

```text
.gitignore
.humanize/kernel-agent/refined-plan.md
README.md
workloads/
src/<task_name>/
bindings/
tests/
benchmarks/
dispatch/
ledgers/attempt-ledger.md
ledgers/optimization-ledger.md
ledgers/lineage.jsonl
ledgers/research-digest.md
ledgers/tuning-decisions.md
benchmarks/performance-map.json
profile-artifacts/README.md
```

它不是固定 ritual，而是 evidence-driven loop：

- prior art 能影响设计时，用 KernelWiki。
- profiler 能改变下一步编辑时，用 ncu-report-skill。
- 不为了完成流程而 profile。
- 不为了完成流程而查知识库。
- 每个工具为什么影响下一步，要记录。

Shape-aware tuning 是一等公民。如果 `W` 是多 regime workload，可以产生多个 specialized kernel 或 dispatch table；如果只是单 shape，就不要过早泛化。

## 6. Prompt 设计值得学习

`kernel-pilot/prompts/` 里的 prompt 很像“实验合同”，不是普通需求描述。

常见字段：

- 远端机器和 container 执行规则。
- 目标 GPU 和 device 选择。
- task name 和 op semantics。
- fixed shape 或 workload distribution。
- baseline 和 comparison target。
- correctness oracle 和 tolerance。
- benchmark 方法、统计量、raw logs。
- 允许的实现来源和 license/source 记录要求。
- promotion criterion。
- target blocked 时的停止条件。
- final report 必须包含什么。

例子：

- B200 `int8_scaled_mm` prompt：单 shape，要求 median latency 相比 SGLang baseline 至少 `2.5x`。
- B200 FA4 MHA prompt：BF16 forward-only attention，要求在 8 个 B200 case 上 geometric-mean TFLOPS 至少超过官方 FlashAttention-4 `5%`。

这类 prompt 的最大价值是防止 benchmark drift：shape、baseline、oracle、promotion rule 在优化前就固定。

## 7. 系统性经验

### 7.1 把证据源和行动循环分开

- KernelWiki 回答“有没有 prior art”。
- ncu-report-skill 回答“profile 具体说明了什么”。
- Humanize/RLCR 回答“这一轮是否足够好”。
- task workspace 负责代码、测试、benchmark、ledger 和最终 promotion。

这样比一个超长 skill 更稳定，也更容易维护。

### 7.2 用 ledger，不靠聊天记忆

kernel tuning 很容易产生大量失败尝试。KernelPilot 用这些文件外化记忆：

- `attempt-ledger.md`：所有尝试，包括失败。
- `optimization-ledger.md`：正确且有测量收益的候选。
- `lineage.jsonl`：candidate 父子关系。
- `research-digest.md`：prior-art 来源和影响。
- `tuning-decisions.md`：shape dispatch / autotune 决策。

这对跨 session、跨 agent 的长实验很关键。

### 7.3 profiling 不是仪式

只有 profile 能改变下一步时才 profile。

适合触发 NCU 的情况：

- baseline bottleneck 不清楚。
- 正确候选 regress 或 plateau。
- 候选异常快或异常慢。
- 下一步依赖于区分 memory、occupancy、scheduling、tensor-core utilization、tail effect。
- reviewer 要求 profiler evidence。

### 7.4 prior art 是证据，不是权威

KernelWiki 提供 PR、artifact、doc、confidence 和 source trace，但最终 promotion 仍然看本地 correctness 和 benchmark。

复制来的技巧只有通过当前任务的 `K/R/W` 合同后才算有效。

### 7.5 kernel 工作需要小 workspace

KDA 和 KernelPilot 都强调 standalone workspace。原因很实际：

- 大型 framework repo 状态太复杂。
- standalone harness 更容易构建。
- candidate 更容易比较。
- benchmark 和 profile artifacts 更容易管理。
- 最后也更容易判断成果应该 upstream、patch framework，还是留作实验。

## 8. 对 ROCm 思考的间接启发

本文不讨论并入 `rocm3d`，但这些设计明显可迁移：

- 用 `K/R/W` 快速定义 kernel 优化任务。
- 用 standalone workspace 做 HIP/Triton/AITER/CK 实验。
- 用 ledger 替代聊天记忆。
- prior-art lookup 必须可引用、可追溯。
- profiling 要 decision-triggered，而不是 ritual-triggered。
- shape distribution 决定是否需要 dispatch/autotune。
- promotion criterion 同时包含 correctness、workload latency 和 reproducibility。

不可直接迁移的是 NVIDIA-specific 内容：

- Nsight Compute 指标和 B200 metric names。
- SM100 特性：`tcgen05`、TMEM、CLC、NVFP4。
- CuTe / CUTLASS / FlashAttention-4 路径。

ROCm 对应层需要换成 TraceLens、rocprof、AITER/ATOM、CK、hipBLASLt/rocBLAS、MIOpen 和 ROCm library forks。

## 9. 从 AMD 高性能 kernel 设计角度看 rocm-kernels-for-3d

如果只看完整 perf 闭环，`rocm-kernels-for-3d` 本来就不应该单独承担全部职责；它需要和 `rocm-perf-analysis`、`rocm-lib-compat` 等 skill 配合。这里评估的是更窄的问题：它作为“指导 agent 在 AMD GPU 上写出高性能 kernel / wrapper / kernel-family 优化”的 skill 是否充分。

结论：对“把已有 AITER/ATOM/ROCm library 能力正确接入 3D/VLA/WM repo”来说，当前 `rocm-kernels-for-3d` 已经比较充分；但对“指导 agent 从零或半从零写出 AMD GPU 高性能 kernel”来说，还不充分。它现在更像 **AITER/ATOM wrapper + kernel-family routing skill**，不是 ROCm 版 KernelWiki / KernelPilot 的“kernel design skill”。

### 9.1 已经充分的部分

当前 `rocm-kernels-for-3d` 的强项是指导 agent 正确使用已有高性能路径：

- 目标边界清楚：只管 3D / Video / WM / VLA / DiT 推理，不混 LLM serving、training、多节点。
- 不盲目 attention-first：明确要求根据 `rocm-perf-analysis` 的热点决定目标。
- AITER/ATOM 接入指导实用：`model_ops/` mirror、loader shuffle、quant path、piecewise compile、`cos_max` 验证，这些都是 agent 很容易踩坑的地方。
- `cookbook.md` 已经给了可复制 skeleton，适合让 agent 快速完成“把高性能库接进模型”的工作。
- 已经吸收了一部分 KernelPilot 思想：standalone workspace、ledger、prior-art lookup、profile only when useful、promotion decision。

所以它适合处理这类任务：

- 把 AITER attention/GEMM/norm/quant/MoE wrapper 接进模型。
- 判断某个热点应该走 AITER、ROCm library backend、model-side layout cleanup，还是 upstream issue。
- 在已有 ATOM/AITER reference 下做可靠迁移和验证。

### 9.2 不足一：缺 ROCm kernel design 知识库层

NVIDIA 侧有 KernelWiki：硬件特性、technique、kernel case、problem pattern、migration、confidence/source trace。ROCm 侧目前 `rocm-kernels-for-3d` 主要引用 AITER/ATOM，但没有系统整理 AMD GPU 的 kernel 设计知识。

缺口包括：

- CDNA3/CDNA4 wavefront、LDS、VGPR、occupancy、MFMA / WMMA / CK tile 的设计约束。
- MI300X / MI350X 常见瓶颈 pattern：LDS bank conflict、VGPR pressure、global load coalescing、MFMA utilization、memory latency hiding。
- CK / AITER / Triton / HIP kernel 什么时候该用哪个。
- conv / sparse conv / raster / gather-scatter 这类非 AITER 路径的高性能设计 pattern。

也就是说，它现在有“怎么接已有高性能 op”，但缺“怎么设计一个 AMD 高性能 kernel”的知识层。

### 9.3 不足二：缺 from-scratch / semi-from-scratch kernel recipe

当前文档明确说“优先用现有维护路径，scratch kernel 是 escape hatch”，这个原则是对的。但如果真的需要写 HIP/Triton kernel，当前 cookbook 不足以指导 agent 做出高性能版本。

缺少的内容包括：

- AMD kernel 的 problem -> cause -> fix 诊断表。
- HIP/Triton/CK 原型如何从 naive 到 performant 的迭代路径。
- MFMA、LDS staging、vectorized load/store、wave-level primitives、occupancy/VGPR tradeoff 的设计模板。
- 不同 shape regime 下何时做 specialization / dispatch / autotune。
- 如何把 rocprof / omniperf / TraceLens 指标映射到下一步 kernel edit。

这部分在 NVIDIA 侧分别由 KernelWiki 和 ncu-report-skill 承担；ROCm 侧目前还没有同等密度的材料。

### 9.4 不足三：缺 ROCm prior-art 索引机制

现在 prior-art lookup 只是建议“查 AITER、ATOM、ROCm forks、PyTorch ROCm issues、upstream PRs”。这对高级人类够用，但对 agent 不够稳定。

理想上需要类似 KernelWiki 的索引层：

- AITER op / PR / example 的问题索引。
- ATOM wrapper / model reference 的路径索引。
- Composable Kernel examples 的 kernel-family 索引。
- Triton AMD、PyTorch ROCm、hipBLASLt、MIOpen 相关 PR 的索引。
- “这个 kernel family 应该查哪些 upstream 文件”的固定入口。

没有这个层，agent 很容易只会重复读 `rocm-kernels-for-3d`，而不是系统性查 prior art。

### 9.5 更准确的定位

当前 `rocm-kernels-for-3d` 的定位可以更精确地写成：

- **充分**：指导 agent 把 AITER/ATOM 已有高性能路径接到 3D/VLA/WM 模型里。
- **基本充分**：指导 agent 判断热点应该走 AITER、library backend、model-side layout cleanup，还是 upstream issue。
- **不足**：指导 agent 设计新的 AMD GPU 高性能 kernel。
- **不足**：沉淀 ROCm 版 KernelWiki 式 prior-art / technique / diagnosis 知识。

### 9.6 已补齐的方式

不建议继续把所有内容堆进 `rocm-kernels-for-3d/SKILL.md` 主体。现在已新增第四个 skill：

- `rocm-kernels-for-3d`：继续负责已有 AITER/ATOM/ROCm-library 高性能路径的接入。
- `rocm-new-kernels`：负责缺失 op 的新 HIP/Triton/CK kernel 设计与迭代。

`rocm-new-kernels` 当前 v0.1 已补入：

- `K/R/W` kernel contract。
- standalone harness / correctness oracle / benchmark / ledger。
- MI300X/MI325X (`gfx942`) 与 MI350X/MI355X (`gfx950`) 的硬件 knowhow。
- CDNA wavefront、LDS、VGPR、memory coalescing、shape specialization 等设计检查点。
- GEAK-inspired candidate generation / reflection / optimization loop。
- profiling 工具分层：宏观 phase/op ranking 复用 `rocm-perf-analysis` + TraceLens；custom kernel op-level 慢因使用 `rocprof v3`；更深 occupancy / memory 分析再用 omniperf。
- MsDeformAttn-style irregular gather kernel pattern。

一句话：`rocm-kernels-for-3d` 对“高性能库接入”足够强；`rocm-new-kernels` 补上“缺失 op 的新 kernel 设计”入口。后续如果要继续增强，应优先补 ROCm 版 KernelWiki / diagnosis / prior-art 索引，而不是继续扩大任一单个 skill。

## 10. 一句话 mental model

```text
KDA:
  通用 evidence loop

KernelWiki:
  带 schema、source、confidence、query tools 的 prior-art 数据库

ncu-report-skill:
  profiler 采集、诊断、报告和下一步建议

KernelPilot:
  把 K/R/W、workspace、ledger、KernelWiki、NCU、RLCR review、
  correctness、benchmark 和 promotion 串成 kernel tuning loop
```

这套生态成熟的地方在于：它把记忆放进文件，把证据放进 ledger，把 prior art 放进可验证 wiki，把 profiler reasoning 放进诊断 skill，把迭代控制放进 review loop。
