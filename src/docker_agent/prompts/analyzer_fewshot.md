# Few-shot examples for install failure analyzer
# Injected as-is into prompt payload key: fewshot_examples.
# Keep to 2-3 high-quality examples; each must show a complete diagnosis + correct patch.

---

## Example 1 — pytorch3d source build fails (torch import at build time)

**Error signal:**
```
ModuleNotFoundError: No module named 'torch'
ERROR: Failed to build 'git+https://github.com/facebookresearch/pytorch3d.git'
Getting requirements to build wheel: finished with status 'error'
```

**Root cause:** pip's default PEP 517 build isolation creates a clean subprocess without the conda env's torch, so setup.py that does `import torch` at build time always fails.

**Correct fix — replace the source build with the pre-built ROCm 6.4 / Python 3.12 wheel BEFORE `pip install -r requirements.txt`:**
```json
{
  "action": "patch_script",
  "patches": [
    {
      "op": "prepend_block",
      "target": "install_script",
      "match": "pip install -r requirements.txt",
      "content": "# Install pytorch3d from pre-built ROCm 6.4 / Python 3.12 wheel\npip install https://github.com/ZJLi2013/pytorch3d/releases/download/rocm6.4-py3.12/pytorch3d-0.7.9-cp312-cp312-linux_x86_64.whl"
    }
  ]
}
```

**Do NOT skip** these packages with `[SKIP]` comments — that silently breaks downstream imports.

---

## Example 2 — git+URL entry in requirements.txt bypasses EXCLUDE_PKGS grep

**Error signal:**
```
Collecting git+https://github.com/nerfstudio-project/gsplat.git (from -r requirements.txt)
ModuleNotFoundError: No module named 'torch'
ERROR: Failed to build 'git+https://github.com/nerfstudio-project/gsplat.git'
```

**Root cause:** The install script already installed gsplat from the AMD prebuilt index, but `requirements.txt` also contains a `git+URL` line for the same package. The grep exclusion only strips package-name form lines (e.g. `gsplat>=1.0`), not URL-form lines starting with `git+https://`.

**Correct fix — prepend a git+URL strip pass before `pip install -r requirements.txt`:**
```json
{
  "action": "patch_script",
  "patches": [
    {
      "op": "prepend_block",
      "target": "install_script",
      "match": "pip install -r requirements.txt",
      "content": "# Strip git+URL entries for packages already installed via AMD wheel or pre-built wheel\ngrep -vEi \"git\\+https?://[^[:space:]]*(gsplat|pytorch3d|xformers|flash.attn|flash-attn|triton)\" requirements.txt > requirements.tmp && mv requirements.tmp requirements.txt"
    }
  ]
}
```

---

## Example 3 — pyproject.toml pins CUDA packages; `pip install -e .` reinstalls wrong version

**Error signal:**
```
Installing collected packages: torch
  Attempting uninstall: torch
    Found existing installation: torch 2.8.0+rocm6.4
Successfully installed torch-2.6.0+cu124
```
or:
```
ERROR: Failed to build installable wheels for some pyproject.toml based builds
ModuleNotFoundError: No module named 'torch'
  error: subprocess-exited-with-error (building pytorch3d)
```

**Root cause:** `pyproject.toml` lists ROCm-sensitive packages (e.g. `"torch>=2.0"`, `"pytorch3d @ git+https://..."`) in `[project.dependencies]`. When `pip install -e .` (Block F) resolves dependencies, it re-installs the default CUDA wheel from PyPI or tries to source-build pytorch3d — overwriting the ROCm wheels already installed in Block C/D.

**Correct fix — strip Block C/D packages from pyproject.toml BEFORE `pip install -e .`:**
```json
{
  "action": "patch_script",
  "patches": [
    {
      "op": "prepend_block",
      "target": "install_script",
      "match": "pip install -e .",
      "content": "# Strip ROCm-sensitive package pins from pyproject.toml before editable install\npython3 -c \"\nimport re, pathlib; p = pathlib.Path('pyproject.toml')\npkg = r'torch|torchvision|torchaudio|xformers|gsplat|flash\\.attn|flash-attn|triton|torch\\.geometric|pyg\\.lib|torch\\.scatter|torch\\.sparse|torch\\.cluster|torch\\.spline\\.conv|pytorch3d'\npat1 = re.compile(r'[ \\t]*\\\"(' + pkg + r')[^\\\"]*\\\",?[^\\n]*\\n', re.I)\npat2 = re.compile(r'[ \\t]*(' + pkg + r')\\s*=\\s*\\{[^\\}]*git[^\\}]*\\}[^\\n]*\\n', re.I)\ntxt = p.read_text(); txt = pat1.sub('', txt); txt = pat2.sub('', txt); p.write_text(txt)\nprint('pyproject.toml: ROCm-sensitive pins removed')\n\""
    }
  ]
}
```

**Key rule:** always run pyproject.toml stripping before `pip install -e .`, regardless of whether `requirements.txt` was also patched.
