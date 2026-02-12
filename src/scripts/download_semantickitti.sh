#!/bin/bash

set -euo pipefail

# SemanticKITTI download + extract (HPC-friendly)
#
# This wraps `src/dataset/download_semantickitti.py` so you can run a single command on a cluster login node.
# Defaults to writing into: repo_root/data/dataset/semantickitti

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" "$REPO_ROOT/src/dataset/download_semantickitti.py" "$@"

cat <<'EOF'

Expected structure:
  data/dataset/semantickitti/dataset/sequences/00/velodyne/*.bin
  data/dataset/semantickitti/dataset/sequences/00/labels/*.label

Tip (disk space): `--delete-zips` removes the large zip(s) after extracting.
EOF
