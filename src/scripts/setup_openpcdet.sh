#!/bin/bash
# Setup OpenPCDet in the lidarcomp311 environment and download PointPillar KITTI checkpoint.
#
# Usage:
#   bash src/scripts/setup_openpcdet.sh
#
# Optional env:
#   CONDA_ENV=lidarcomp311
#   OPENPCDET_ROOT=third_party/OpenPCDet
#   INSTALL_CKPT=1
#   OPENPCDET_PP18M_URL=https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing
#   OPENPCDET_PP18M_CKPT=data/checkpoints/openpcdet_pointpillar_18M.pth

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/018219422/lidar_pointcloud_compression}
cd "${ROOT_DIR}"

CONDA_ENV=${CONDA_ENV:-lidarcomp311}
CONDA_PREFIX_DIR="${HOME}/miniconda3/envs/${CONDA_ENV}"
if [[ ! -x "${CONDA_PREFIX_DIR}/bin/python" ]]; then
  echo "Error: expected python not found: ${CONDA_PREFIX_DIR}/bin/python" >&2
  exit 1
fi
PYTHON_BIN="${CONDA_PREFIX_DIR}/bin/python"
PIP_BIN="${CONDA_PREFIX_DIR}/bin/pip"

OPENPCDET_ROOT=${OPENPCDET_ROOT:-third_party/OpenPCDet}
INSTALL_CKPT=${INSTALL_CKPT:-1}
OPENPCDET_PP18M_URL=${OPENPCDET_PP18M_URL:-https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing}
OPENPCDET_PP18M_CKPT=${OPENPCDET_PP18M_CKPT:-data/checkpoints/openpcdet_pointpillar_18M.pth}

if [[ ! -d "${OPENPCDET_ROOT}" ]]; then
  git clone https://github.com/open-mmlab/OpenPCDet.git "${OPENPCDET_ROOT}"
fi

# CUDA 11.8 toolchain paths on this cluster.
NVHPC_ROOT=${NVHPC_ROOT:-/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11}
CUDA_HOME=${CUDA_HOME:-${NVHPC_ROOT}/cuda/11.8}
CUDA_MATH_INC=${CUDA_MATH_INC:-${NVHPC_ROOT}/math_libs/11.8/targets/x86_64-linux/include}
CUDA_MATH_LIB=${CUDA_MATH_LIB:-${NVHPC_ROOT}/math_libs/11.8/targets/x86_64-linux/lib}
TORCH_LIB="$("${PYTHON_BIN}" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${TORCH_LIB}:${CUDA_HOME}/lib64:${CUDA_MATH_LIB}:${LD_LIBRARY_PATH:-}"
export C_INCLUDE_PATH="${CUDA_MATH_INC}:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="${CUDA_MATH_INC}:${CPLUS_INCLUDE_PATH:-}"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-6.0;8.0}
export MAX_JOBS=${MAX_JOBS:-2}

"${PIP_BIN}" install -r "${OPENPCDET_ROOT}/requirements.txt"
"${PIP_BIN}" install gdown
(cd "${OPENPCDET_ROOT}" && "${PIP_BIN}" install -e . --no-build-isolation)

if [[ "${INSTALL_CKPT}" == "1" ]]; then
  mkdir -p "$(dirname "${OPENPCDET_PP18M_CKPT}")"
  if [[ ! -f "${OPENPCDET_PP18M_CKPT}" ]]; then
    "${PYTHON_BIN}" -m gdown --fuzzy "${OPENPCDET_PP18M_URL}" -O "${OPENPCDET_PP18M_CKPT}"
  fi
fi

# Import check for core CUDA extensions.
"${PYTHON_BIN}" -c "from pcdet.ops.iou3d_nms import iou3d_nms_cuda; from pcdet.ops.roiaware_pool3d import roiaware_pool3d_cuda; from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_stack_cuda; print('OpenPCDet setup OK')"

echo "OpenPCDet root: ${OPENPCDET_ROOT}"
echo "PointPillar ckpt: ${OPENPCDET_PP18M_CKPT}"
