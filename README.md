# rocm3d-autorun

Cursor agent skill for porting ML repos (3D generation, reconstruction, world models, video generation, etc.) to AMD ROCm.

Provides a canonical **ROCm library replacement table** — when you encounter a CUDA-dependent library
in a repo's dependencies, the skill tells Cursor exactly how to install the ROCm equivalent.

## Usage

In Cursor, invoke the skill:

```
"使用 rocm-lib-compat skill，给 https://github.com/<owner>/<repo> 生成 ROCm install 脚本"
```

## Supported Repos

The following repos have been verified on AMD MI300X with ROCm:

### 3D Generation & Reconstruction

| Repo | Domain | Key ROCm Libs | Status |
|------|--------|---------------|--------|
| [wgsxm/PartCrafter](https://github.com/wgsxm/PartCrafter) | Part-aware 3D generation | pytorch3d | ✅ Verified |
| [apple/ml-sharp](https://github.com/apple/ml-sharp) | 3D reconstruction | gsplat | ✅ Verified |
| [openai/shap-e](https://github.com/openai/shap-e) | Text/image to 3D | — | ✅ Verified |
| [naver/dust3r](https://github.com/naver/dust3r) | Dense stereo reconstruction | croco (ext build) | ✅ Verified |
| [facebookresearch/fast3r](https://github.com/facebookresearch/fast3r) | Fast 3D reconstruction | croco (ext build) | ✅ Verified |
| [nv-tlabs/Difix3D](https://github.com/nv-tlabs/Difix3D) | 3D diffusion fixing | xformers | ✅ Verified |
| [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | Visual grounding | — | ✅ Verified |
| [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | Monocular depth + 3DGS | xformers, gsplat | ✅ Verified |
| [expenses/gaussian-splatting](https://github.com/expenses/gaussian-splatting) | 3DGS (ROCm fork) | diff-gaussian-rasterization | ✅ Verified |
| [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) | Map reconstruction | — | ✅ Verified |

### 3D/4D Generation (AI-generated scripts)

| Repo | Domain | Key ROCm Libs | Status |
|------|--------|---------------|--------|
| [fudan-zvg/4d-gaussian-splatting](https://github.com/fudan-zvg/4d-gaussian-splatting) | 4D Gaussians | diff-gaussian-rasterization, simple-knn | ✅ Script generated |
| [VITA-Group/Anything-3D](https://github.com/VITA-Group/Anything-3D) | Anything to 3D | — | ✅ Script generated |
| [any4d](https://github.com/) | 4D generation | — | ✅ Script generated |
| [DimensionX](https://github.com/) | Multi-dim generation | — | ✅ Script generated |
| [nv-tlabs/FLARE](https://github.com/nv-tlabs/FLARE) | Face generation | pytorch3d | ✅ Script generated |
| [Gen3C](https://github.com/) | 3D-consistent generation | — | ✅ Script generated |
| [mv-inverse](https://github.com/) | Multi-view inverse | — | ✅ Script generated |
| [jiangzhongshi/RecamMaster](https://github.com/jiangzhongshi/RecamMaster) | Camera re-rendering | — | ✅ Script generated |

### Video Generation / World Models

| Repo | Domain | Key ROCm Libs | Status |
|------|--------|---------------|--------|
| [SkyworkAI/Matrix-Game](https://github.com/SkyworkAI/Matrix-Game) | Video world model | flash-attn → **AITER CK** | ✅ Verified (PR ready) |

## Project Structure

```
.cursor/skills/rocm-lib-compat/
  SKILL.md       # Core skill — ROCm lib replacement table + AITER FA3
```

## Core Replacement Table (highlights)

| Library | ROCm Solution |
|---------|--------------|
| flash-attn (ROCm 6.x) | `pip install flash-attn --index-url=https://pypi.amd.com/simple` (Triton) |
| flash-attn (ROCm 7.x) | `pip install aiter` — AITER CK backend, **~25% faster** |
| xformers | `pip install xformers --index-url https://download.pytorch.org/whl/rocm6.4` |
| gsplat | `pip install gsplat --index-url=https://pypi.amd.com/simple` |
| pytorch3d | Pre-built ROCm wheel |

See [`.cursor/skills/rocm-lib-compat/SKILL.md`](.cursor/skills/rocm-lib-compat/SKILL.md) for the full table, AITER integration patterns, and troubleshooting guide.

## Contributing

For new ROCm library mappings: update `.cursor/skills/rocm-lib-compat/SKILL.md`
