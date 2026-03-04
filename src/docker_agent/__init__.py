"""
docker_agent MVP package.

入口：python -m docker_agent（执行 __main__.py）。
示例：
  PYTHONPATH=./src python3 -m docker_agent --repo_url https://github.com/facebookresearch/vggt

PowerShell:
  $env:PYTHONPATH = ".\\src"
  python -m docker_agent --repo_url https://github.com/facebookresearch/vggt
"""
from .agent import BuildRequest, BuildResult, DockerAgent

__all__ = ["DockerAgent", "BuildRequest", "BuildResult"]
