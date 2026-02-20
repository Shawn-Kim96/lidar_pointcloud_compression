#!/bin/bash
# Download PointPillars checkpoint on login node before GPU training.
# Default source: zhulf0804/PointPillars pretrained/epoch_160.pth
#
# Usage:
#   bash src/scripts/download_pointpillar_checkpoint.sh
#   POINTPILLAR_CKPT_PATH=data/checkpoints/pointpillars_epoch_160.pth bash src/scripts/download_pointpillar_checkpoint.sh
#   POINTPILLAR_CKPT_URL=<custom_url> bash src/scripts/download_pointpillar_checkpoint.sh

set -euo pipefail

DEST_PATH=${POINTPILLAR_CKPT_PATH:-data/checkpoints/pointpillars_epoch_160.pth}
mkdir -p "$(dirname "$DEST_PATH")"

if [[ -s "$DEST_PATH" ]]; then
  echo "[download_pointpillar_checkpoint] Already exists: $DEST_PATH"
  exit 0
fi

TMP_PATH="${DEST_PATH}.tmp"
rm -f "$TMP_PATH"

URLS=(
  "${POINTPILLAR_CKPT_URL:-}"
  "https://raw.githubusercontent.com/zhulf0804/PointPillars/main/pretrained/epoch_160.pth"
)

try_download_with_curl() {
  local url="$1"
  if [[ -z "$url" ]]; then
    return 1
  fi
  if ! command -v curl >/dev/null 2>&1; then
    return 1
  fi
  echo "[download_pointpillar_checkpoint] Trying curl URL: $url"
  if curl -L --fail --retry 3 --retry-delay 2 "$url" -o "$TMP_PATH"; then
    return 0
  fi
  return 1
}

ok=0
for u in "${URLS[@]}"; do
  if try_download_with_curl "$u"; then
    ok=1
    break
  fi
  rm -f "$TMP_PATH"
done

if [[ "$ok" -ne 1 ]]; then
  echo "[download_pointpillar_checkpoint] Failed to download checkpoint."
  echo "Set POINTPILLAR_CKPT_URL explicitly and retry."
  exit 1
fi

size_bytes=$(stat -c%s "$TMP_PATH" 2>/dev/null || echo 0)
if [[ "$size_bytes" -lt 1000000 ]]; then
  echo "[download_pointpillar_checkpoint] Downloaded file is unexpectedly small (${size_bytes} bytes)."
  rm -f "$TMP_PATH"
  exit 1
fi

mv "$TMP_PATH" "$DEST_PATH"
echo "[download_pointpillar_checkpoint] Saved: $DEST_PATH (${size_bytes} bytes)"
