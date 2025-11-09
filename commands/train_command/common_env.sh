#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

DEFAULT_CUDA_MODULE=${DEFAULT_CUDA_MODULE:-cuda/12.4}
DEFAULT_CONDA_HOME=${DEFAULT_CONDA_HOME:-"$HOME/miniconda3"}
DEFAULT_CONDA_ENV=${DEFAULT_CONDA_ENV:-myenv}

function init_environment() {
  local cuda_module="${1:-$DEFAULT_CUDA_MODULE}"
  local conda_home="${2:-$DEFAULT_CONDA_HOME}"
  local conda_env="${3:-$DEFAULT_CONDA_ENV}"

  module load "${cuda_module}"
  # shellcheck source=/dev/null
  source "${conda_home}/bin/activate" "${conda_env}"

  export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
  export PYTORCH_NO_NVML=${PYTORCH_NO_NVML:-1}
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^docker0,lo}
  export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5}
  export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
  export TORCH_USE_CUDA_DSA=${TORCH_USE_CUDA_DSA:-1}

  cd "${PROJECT_ROOT}"
  mkdir -p printout/output_file printout/error_file
}

function ensure_head_node_ip() {
  local head_node
  head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  getent hosts "$head_node" | awk '{ print $1 }'
}

function timestamp() {
  date +"%Y%m%d_%H%M%S"
}


