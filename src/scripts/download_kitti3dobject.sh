#!/bin/bash

set -euo pipefail

# KITTI 3D Object download + extract wrapper
#
# Default destination:
#   repo_root/data/dataset/kitti3dobject
#
# Examples:
#   bash src/scripts/download_kitti3dobject.sh
#   bash src/scripts/download_kitti3dobject.sh --only all --extractor unzip --delete-zips

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" "$REPO_ROOT/src/dataset/download_kitti3dobject.py" "$@"

