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

# Define random seeds
seeds=(42)  # Add seeds

model_name="Meta-Llama-3-8B"
dataset_name="GSM8K_KTO"

# Loop over each seed
for seed in "${seeds[@]}"; do
    echo "Running with seed ${seed}"

    # Set output directory for this seed
    output_dir="saves/${model_name}/full/${dataset_name}_full_KTO_${seed}"

    # Run the training with all parameters specified
    srun --ntasks=1 --gres=gpu:8 --ntasks-per-node=1 llamafactory-cli train \
        --model_name_or_path checkpoints/${model_name} \
        --stage kto \
        --pref_beta 0.1\
        --do_train \
        --finetuning_type full \
        --deepspeed examples/deepspeed/ds_z3_config.json \
        --dataset ${dataset_name} \
        --template llama3 \
        --cutoff_len 2048 \
        --overwrite_cache \
        --preprocessing_num_workers 32 \
        --output_dir ${output_dir} \
        --logging_steps 100 \
        --save_steps 100 \
        --plot_loss \
        --overwrite_output_dir \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1.0e-5 \
        --num_train_epochs 5 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.1 \
        --bf16 \
        --ddp_timeout 180000000 \
        --val_size 0.1 \
        --per_device_eval_batch_size 32 \
        --eval_strategy steps \
        --eval_steps 100 \
        --seed ${seed} \
        >> "printout/output_file/output_${current_time}_${seed}.out" \
        2>> "printout/error_file/error_${current_time}_${seed}.err"
done
