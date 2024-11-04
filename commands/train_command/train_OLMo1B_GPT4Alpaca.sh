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

# Add recommended NCCL settings
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3

# Navigate to your project directory
cd /mbz/users/liyuan/LLaMA-Factory

# Set up current time
current_time=$(date +"%Y%m%d_%H%M%S")

# Define the list of random seeds
seeds=(12345)  # Add as many seeds as you want

model_name="OLMo-1B"
dataset_name="GPT4_alpaca"

# Loop over each seed
for seed in "${seeds[@]}"; do
    echo "Running with seed ${seed}"

    # Set output directory for this seed
    output_dir="saves/${model_name}/full/${dataset_name}_full_sft_${seed}"

    # Run the training with all parameters specified
    srun --ntasks=1 --gres=gpu:8 --ntasks-per-node=1 llamafactory-cli train \
        --model_name_or_path checkpoints/${model_name} \
        --stage sft \
        --do_train \
        --finetuning_type full \
        --deepspeed examples/deepspeed/ds_z3_config.json \
        --dataset ${dataset_name} \
        --template olmo \
        --cutoff_len 2048 \
        --overwrite_cache \
        --preprocessing_num_workers 32 \
        --output_dir ${output_dir} \
        --logging_steps 1000 \
        --save_steps 1000 \
        --plot_loss \
        --overwrite_output_dir \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --learning_rate 1.0e-7 \
        --num_train_epochs 20 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.1 \
        --bf16 \
        --ddp_timeout 180000000 \
        --val_size 0.01 \
        --per_device_eval_batch_size 32 \
        --eval_strategy steps \
        --eval_steps 500 \
        --seed ${seed} \
        >> "printout/output_file/output_${current_time}_${seed}.out" \
        2>> "printout/error_file/error_${current_time}_${seed}.err"
done
