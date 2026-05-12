#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv-jetson}"

if ! "$PYTHON_BIN" - <<'PY' >/dev/null
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)
PY
then
  echo "Jetson CUDA PyTorch wheels on this image are installed for Python 3.10." >&2
  echo "Set PYTHON_BIN to a Python 3.10 interpreter." >&2
  exit 1
fi

rm -rf "$VENV_DIR"

if "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR" 2>/tmp/vrs-venv.err; then
  :
else
  echo "python -m venv failed; falling back to virtualenv." >&2
  cat /tmp/vrs-venv.err >&2 || true
  "$PYTHON_BIN" -m pip install --user virtualenv
  "$PYTHON_BIN" -m virtualenv -p "$PYTHON_BIN" --system-site-packages "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r requirements-jetson.txt
"$VENV_DIR/bin/python" -m pip install --ignore-requires-python --no-deps -e .

"$VENV_DIR/bin/python" - <<'PY'
import sys
import torch

print(f"python {sys.version.split()[0]}")
print(f"torch {torch.__version__}")
print(f"cuda {torch.cuda.is_available()}")
print(f"cuda_version {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"device {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit("Jetson torch was imported, but CUDA is not available")
PY
