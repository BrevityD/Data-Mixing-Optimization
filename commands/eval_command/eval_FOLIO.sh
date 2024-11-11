#!/bin/bash
#SBATCH --partition=mbzuai
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive

# Load necessary modules            
module load cuda/12.1      

# Activate your Conda environment
source ~/miniconda3/bin/activate myenv

# Set NCCL to debug only errors
export NCCL_DEBUG=WARN

# Disable NVML if not required
export PYTORCH_NO_NVML=1
# Add recommended NCCL settings for multi-node
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3

# Navigate to your project directory
cd /mbz/users/liyuan/LLaMA-Factory

# Set up output and error file paths with current time
current_time=$(date +"%Y%m%d_%H%M%S")
seeds=(42)
model_name="FOLIO_like_data_sft_1.0e-6"
# model_name="Meta-Llama-3.1-8B-Instruct"

eval_dataset="FOLIO_validation"
# --model_name_or_path "checkpoints/${model_name}" \
# --model_name_or_path "saves/Meta-Llama-3.1-8B/full/${model_name}" \
# Loop over each seed
for seed in "${seeds[@]}"; do
    echo "Running with seed ${seed}"

    # Run the training with command-line arguments
    srun --ntasks=1 --gres=gpu:8 --ntasks-per-node=1 llamafactory-cli train \
        --model_name_or_path "saves/Meta-Llama-3.1-8B/full/${model_name}" \
        --stage sft \
        --do_predict \
        --finetuning_type full \
        --eval_dataset ${eval_dataset} \
        --template llama3 \
        --cutoff_len 2048 \
        --overwrite_cache \
        --preprocessing_num_workers 32 \
        --output_dir "/mbz/users/liyuan/LLaMA-Factory/results/${eval_dataset}/${model_name}" \
        --overwrite_output_dir \
        --per_device_eval_batch_size 64 \
        --predict_with_generate \
        --seed ${seed} \
        >> "printout/output_file/output_eval_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
        2>> "printout/error_file/error_eval_${SLURM_JOB_ID}_${current_time}_${seed}.err"
done