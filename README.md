# rocm3d-autorun

Automated verification and skill-based migration of ML repos to the ROCm GPU platform.

Given a GitHub repo and a base ROCm Docker image, `rocm3d-autorun` automatically:

1. **Generates** an `install.sh` + `run.sh` for the target repo (via `docs/skills/`)
2. **Executes** install + run inside Docker, capturing structured JSON output
3. **Analyzes** failures with an LLM and auto-patches the install script (retry loop)
4. **Accumulates** fix patterns as reusable experience (`docs/skills/`, `prompts/analyzer_fewshot.md`)

## Supported domains

| Domain | Status |
|--------|--------|
| 3D Generation & Reconstruction | ✅ Active (mvinverse, Anything-3D, any4d, dimensionx, flare, recammaster) |
| 3D Gaussian Splatting acceleration libs | 🔄 Planned |
| World Model | 🔄 Planned |
| VLA (Vision-Language-Action) | 🔄 Planned |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure LLM provider

```bash
cp .env.example .env
# Edit .env — pick one of: openai / anthropic / openai_compat (vLLM, AMD gateway, etc.)
```

### 3. Generate install script for a repo

```bash
# In Cursor / Claude Code — invoke the skill:
# "给 https://github.com/<owner>/<repo> 生成 ROCm install 脚本"
# Output: samples/auto_gen/<repo>_install.sh
```

### 4. Run install + auto-patch loop on a remote GPU node

```bash
git pull
PYTHONPATH=./src python -m docker_agent \
  --repo_url https://github.com/<owner>/<repo> \
  --base-image rocm/pytorch:latest \
  --install-script samples/auto_gen/<repo>_install.sh \
  --auto-patch-on-fail \
  --max-auto-patch-retries 3 \
  -o samples/auto_gen/test_output/<repo>.json
```

### 5. Batch install verification

```bash
bash samples/run_all_install_tests.sh
```

## LLM Provider

`rocm3d-autorun` uses a lightweight, **dependency-free** LLM client (`src/llms/`).  
No LangChain or vendor SDKs required.

Supported providers — set via `.env`:

| `LLM_PROVIDER` | Use case | Auth |
|---|---|---|
| `openai` | OpenAI API | `LLM_API_KEY=sk-...` |
| `anthropic` | Anthropic Claude | `LLM_API_KEY=sk-ant-...` |
| `openai_compat` | AMD Gateway, vLLM, Ollama, Azure | `LLM_API_KEY` + `LLM_BASE_URL` |

See `.env.example` for complete configuration examples.

## Architecture

```
User: repo URL
      ↓
┌──────────────────────────────────────────────────┐
│  skill-1: rocm-install-script-generator          │
│  docs/skills/rocm-install-script-generator/      │
│  Consumer: Cursor / Claude Code agent            │
│                                                  │
│  Reads repo README + requirements,               │
│  assembles install.sh + run.sh per Block A–H.    │
└──────────────────────────────────────────────────┘
      ↓ script failure
┌──────────────────────────────────────────────────┐
│  auto-patcher: llm_log_analyzer                  │
│  src/docker_agent/llm_log_analyzer.py            │
│  Consumer: docker_agent orchestrator (auto)      │
│                                                  │
│  Failure analysis + script patch:                │
│  · analyzer_fewshot.md: error → patch examples  │
│  · analyzer_constraints.txt: patch rules         │
└──────────────────────────────────────────────────┘
```

**Experience accumulation:**

| Type | Location | Consumer |
|------|----------|----------|
| New ROCm compat rules | `docs/skills/rocm-install-script-generator/SKILL.md` | Agent (script gen) |
| Error → patch examples | `src/docker_agent/prompts/analyzer_fewshot.md` | `llm_log_analyzer` |

## Directory Layout

```
src/docker_agent/     Core execution & analysis (install → run → LLM patch loop)
src/llms/             Lightweight LLM client (zero deps, multi-provider)
samples/auto_gen/     AI-generated install/run scripts + test outputs
samples/manually_scripts/  Hand-written reference scripts
docs/skills/          Agent skills for script generation
tools/                Offline analysis helpers
tests/                Unit + integration tests
```

## Requirements

| Dependency | Purpose | Where needed |
|---|---|---|
| Python 3.10+ | Run docker_agent | Remote GPU node |
| Docker + ROCm driver | Container execution | Remote GPU node |
| `pip install -r requirements.txt` | docker-py | Remote GPU node |
| LLM API key (any provider) | Failure analysis + auto-patch | Optional; falls back to `need_human` |

## Current GPU Nodes (MI300)

See project README or team docs for active node hostnames.

## Roadmap

See [Roadmap](README.md#roadmap) for upcoming domain support and planned phases.

---

## Contributing

1. Fork the repo and create a feature branch.
2. For new ROCm compat rules: update `docs/skills/rocm-install-script-generator/SKILL.md`.
3. For new error→patch patterns: append to `src/docker_agent/prompts/analyzer_fewshot.md`.
4. Submit a PR with test evidence (JSON output from `docker_agent`).
