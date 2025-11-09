#!/bin/bash
#SBATCH --partition=highprio
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive

module load cuda/12.4

source "${HOME}/miniconda3/bin/activate" myenv

export NCCL_DEBUG=WARN

export PYTORCH_NO_NVML=1

export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"

cd "${PROJECT_ROOT}"

mkdir -p "${PROJECT_ROOT}/printout/output_file"
mkdir -p "${PROJECT_ROOT}/printout/error_file"

python3 "${SCRIPT_DIR}/load_data.py" \
  >> "${PROJECT_ROOT}/printout/output_file/output_${SLURM_JOB_ID:-manual}.out" \
  2>> "${PROJECT_ROOT}/printout/error_file/error_${SLURM_JOB_ID:-manual}.err"