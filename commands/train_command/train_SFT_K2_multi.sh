#!/bin/bash
#SBATCH --partition=highprio
#SBATCH --time=100:00:00
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --exclude=g42-h100-instance-075,g42-h100-instance-078,g42-h100-instance-079

# Load necessary modules
module load cuda/12.1

# Activate your Conda environment
source ~/miniconda3/bin/activate myenv

# Disable NVML if not required
export PYTORCH_NO_NVML=1
# Add recommended NCCL settings
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
# export NCCL_P2P_LEVEL=NVL

# Navigate to your project directory
cd /mbz/users/liyuan/LLaMA-Factory
mkdir -p printout/output_file
mkdir -p printout/error_file

# Set up current time
current_time=$(date +"%Y%m%d_%H%M%S")

# Define head_node_ip using SLURM's environment variables
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
head_node_ip=$(getent hosts "$head_node" | awk '{ print $1 }')

# Verify head_node_ip
echo "Head node: $head_node ($head_node_ip)"

# Define random seeds
seeds=(42)  # Add more seeds if needed

model_name="K2"
# dataset_name="alpaca_double"
# gsm8k_train,
dataset_name="tulu3-sft"
# dataset_name="flan-v2"
lr=1.0e-6

#   --preprocessing_num_workers 32 \
# Loop over each seed
for seed in "${seeds[@]}"; do
    echo "Running with seed ${seed}"

    # Set output directory for this seed
    output_dir="saves/${model_name}/full/${model_name}_${dataset_name}_sft_${lr}_seed${seed}"

    # Launch training using srun and torchrun
    srun torchrun \
        --nproc_per_node=8 \
        --nnodes=32 \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${head_node_ip}:29500 \
        src/train.py \
        --model_name_or_path checkpoints/${model_name} \
        --stage sft \
        --do_train \
        --finetuning_type full \
        --deepspeed "examples/deepspeed/ds_z3_config.json" \
        --dataset ${dataset_name} \
        --template llama3 \
        --cutoff_len 4096 \
        --overwrite_cache \
        --streaming True \
        --output_dir ${output_dir} \
        --logging_steps 150 \
        --save_steps 150 \
        --eval_steps 150 \
        --plot_loss \
        --overwrite_output_dir \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 2 \
        --learning_rate ${lr} \
        --num_train_epochs 10 \
        --max_steps  3000 \
        --dispatch_batches False \
        --lr_scheduler_type linear \
        --warmup_ratio 0.03 \
        --bf16 \
        --ddp_timeout 180000000 \
        --per_device_eval_batch_size 8 \
        --eval_strategy steps \
        --eval_dataset val \
        --load_best_model_at_end True \
        --save_total_limit 1 \
        --seed "${seed}" \
        >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
        2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"
done
