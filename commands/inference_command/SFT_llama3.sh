#!/bin/bash
#SBATCH --partition=highprio
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive

# Load necessary modules
module load cuda/12.4

# Activate your Conda environment
source ~/miniconda3/bin/activate myenv

# Set NCCL to debug only errors
export NCCL_DEBUG=WARN

# Disable NVML if not required
export PYTORCH_NO_NVML=1

# Add recommended NCCL settings
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3

# Navigate to your project directory
cd /mbz/users/liyuan/LLaMA-Factory

# Set up current time
current_time=$(date +"%Y%m%d_%H%M%S")

# Define random seeds
seed=42

# Define output directory and model details
output_dir="results/data_mixing/exp1"
model_prefix="Llama-3.2-3B"
dataset="exp1_val"
# --model_name_or_path /mbz/users/liyuan/LLaMA-Factory/saves/data_mixing/exp1/${model_prefix}/instr/${model_prefix}_instr_half_math_one_code_one_2.0e-5_seed42 \
# checkpoints=(80 240 250)

# for checkpoint in "${checkpoints[@]}"; do
srun --ntasks=1 --gres=gpu:8 --ntasks-per-node=1 llamafactory-cli train \
    --model_name_or_path /mbz/users/liyuan/LLaMA-Factory/saves/data_mixing/exp1/Llama-3.2-3B/instr/Llama-3.2-3B_instr_one_math_one_code_one_2.0e-5_seed42/new \
    --stage sft \
    --do_eval True \
    --finetuning_type full \
    --eval_dataset ${dataset} \
    --template llama3 \
    --cutoff_len 4096 \
    --overwrite_cache \
    --output_dir ${output_dir}/${model_prefix}/${dataset}/checkpoint-${checkpoint} \
    --overwrite_output_dir True \
    --per_device_eval_batch_size 16 \
    --predict_with_generate True \
    --ddp_timeout 180000000 \
    --seed ${seed} \
    >> "printout/output_file/output_inference_${SLURM_JOB_ID}.out" \
    2>> "printout/error_file/error_inference_${SLURM_JOB_ID}.err"
# done