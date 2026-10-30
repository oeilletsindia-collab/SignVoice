#!/usr/bin/env bash
set -euo pipefail
cd /workspace/SignVoice
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/.python
export MPLCONFIGDIR=/workspace/.cache/matplotlib
mkdir -p "$UV_CACHE_DIR" "$MPLCONFIGDIR"
command -v uv >/dev/null
command -v ffmpeg >/dev/null
uv python install 3.11.17
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.11.17 .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,11), "Use a Python 3.11 virtual environment"'
uv pip sync --python .venv/bin/python requirements-lock-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip check --python .venv/bin/python
