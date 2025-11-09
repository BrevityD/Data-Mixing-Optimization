#!/bin/bash
#SBATCH --partition=highprio
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive

module load cuda/12.4

source ~/miniconda3/bin/activate myenv

export NCCL_DEBUG=WARN
export PYTORCH_NO_NVML=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

mkdir -p printout/output_file
mkdir -p printout/error_file

python3 "$PROJECT_ROOT/data_preprocess/openmathinstruct2/load_data.py"