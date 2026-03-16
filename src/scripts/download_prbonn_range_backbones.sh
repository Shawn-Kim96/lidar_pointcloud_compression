#!/bin/bash

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

OUTPUT_DIR=${OUTPUT_DIR:-third_party/range_view_checkpoints/prbonn}
UNPACK=${UNPACK:-1}

if [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${ROOT_DIR}/${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

download_file() {
  local url="$1"
  local dst="$2"

  if [[ -s "${dst}" ]]; then
    echo "[track2-download] skip existing ${dst}"
    return 0
  fi

  local tmp="${dst}.part"
  rm -f "${tmp}"

  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --connect-timeout 20 -o "${tmp}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${tmp}" "${url}"
  else
    echo "Error: neither curl nor wget is available for checkpoint download." >&2
    return 1
  fi

  mv "${tmp}" "${dst}"
  echo "[track2-download] downloaded ${dst}"
}

maybe_unpack() {
  local archive="$1"
  local stem="$2"

  if [[ "${UNPACK}" != "1" ]]; then
    return 0
  fi

  local out_dir="${OUTPUT_DIR}/${stem}"
  mkdir -p "${out_dir}"

  if find "${out_dir}" -mindepth 1 -maxdepth 1 | read -r _; then
    echo "[track2-download] skip unpack existing ${out_dir}"
    return 0
  fi

  tar -xzf "${archive}" -C "${out_dir}"
  echo "[track2-download] unpacked ${archive} -> ${out_dir}"
}

DARKNET53_URL="https://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/darknet53.tar.gz"
DARKNET53_1024_URL="https://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/darknet53-1024.tar.gz"

DARKNET53_ARCHIVE="${OUTPUT_DIR}/darknet53.tar.gz"
DARKNET53_1024_ARCHIVE="${OUTPUT_DIR}/darknet53-1024.tar.gz"

download_file "${DARKNET53_URL}" "${DARKNET53_ARCHIVE}"
download_file "${DARKNET53_1024_URL}" "${DARKNET53_1024_ARCHIVE}"

maybe_unpack "${DARKNET53_ARCHIVE}" "darknet53"
maybe_unpack "${DARKNET53_1024_ARCHIVE}" "darknet53-1024"

MANIFEST="${OUTPUT_DIR}/manifest.txt"
cat > "${MANIFEST}" <<EOF
Downloaded: $(date '+%Y-%m-%d %H:%M:%S %Z')
Source 1: ${DARKNET53_URL}
Archive 1: ${DARKNET53_ARCHIVE}
Extracted 1: ${OUTPUT_DIR}/darknet53
Source 2: ${DARKNET53_1024_URL}
Archive 2: ${DARKNET53_1024_ARCHIVE}
Extracted 2: ${OUTPUT_DIR}/darknet53-1024
EOF

echo "[track2-download] manifest=${MANIFEST}"
