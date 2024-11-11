#!/bin/bash
#SBATCH --partition=mbzuai
#SBATCH --time=10:00:00
#SBATCH --nodes=2
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

# Set up distributed environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Navigate to your project directory
cd /mbz/users/liyuan/LLaMA-Factory
mkdir -p printout/output_file
mkdir -p printout/error_file

# Set up current time
current_time=$(date +"%Y%m%d_%H%M%S")

# Define random seeds
seeds=(42)  # Add more seeds if needed

# Model and training parameters
model_name="Meta-Llama-3.1-8B"
dataset_name="FOLIO_like_data"
lr=1.0e-5

NNODES=2 

# Loop over each seed
for seed in "${seeds[@]}"; do
    echo "Running with seed ${seed}"

    # Set output directory for this seed
    output_dir="saves/${model_name}/full/${dataset_name}_sft_${lr}/seed_${seed}"

    for RANK in $(seq 0 $((NNODES - 1))); do
        srun --nodes=1 --ntasks=1 \
            --export=ALL,MASTER_ADDR=$MASTER_ADDR,MASTER_PORT=$MASTER_PORT,NNODES=$NNODES,RANK=$RANK \
            llamafactory-cli train \
            --model_name_or_path checkpoints/${model_name} \
            --stage sft \
            --do_train \
            --finetuning_type full \
            --deepspeed "examples/deepspeed/ds_z3_config.json" \
            --dataset ${dataset_name} \
            --template llama3 \
            --cutoff_len 2048 \
            --overwrite_cache \
            --preprocessing_num_workers 16 \
            --output_dir ${output_dir} \
            --logging_steps 100 \
            --save_steps 1000 \
            --plot_loss \
            --overwrite_output_dir \
            --per_device_train_batch_size 4 \
            --gradient_accumulation_steps 4 \
            --learning_rate ${lr} \
            --num_train_epochs 3 \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.1 \
            --bf16 \
            --ddp_timeout 180000000 \
            --val_size 0.01 \
            --per_device_eval_batch_size 16 \
            --eval_strategy steps \
            --eval_steps 1000 \
            --seed "${seed}" \
            >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
            2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err" &
    done
done
