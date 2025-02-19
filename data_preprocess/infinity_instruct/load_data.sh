#!/bin/bash
#SBATCH --partition=highprio
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive

# Load necessary modules
module load cuda/12.4

# Activate your Conda environment
source ~/miniconda3/bin/activate myenv

# Set NCCL to debug only warnings and errors
export NCCL_DEBUG=WARN

# Disable NVML if not required
export PYTORCH_NO_NVML=1

# Add recommended NCCL settings
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3

# Navigate to your project directory
cd /mbz/users/liyuan/LLaMA-Factory

mkdir -p printout/output_file
mkdir -p printout/error_file


python3 /mbz/users/liyuan/LLaMA-Factory/data_preprocess/infinity_instruct/load_data.py \
>> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"