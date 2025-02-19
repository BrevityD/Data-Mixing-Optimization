#!/bin/bash
#SBATCH --partition=mbzuai
#SBATCH --time=500:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive


# Load necessary modules

module load cuda/12.4
source ~/miniconda3/bin/activate myenv
cd /mbz/users/liyuan/LLaMA-Factory

nvidia-smi
export PYTORCH_NO_NVML=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export TORCH_USE_CUDA_DSA=1

# Navigate to your project directory


mkdir -p printout/output_file
mkdir -p printout/error_file

# Set up current time
current_time=$(date +"%Y%m%d_%H%M%S")

# Define head_node_ip using SLURM's environment variables
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
head_node_ip=$(getent hosts "$head_node" | awk '{ print $1 }')

# Verify head_node_ip
echo "Head node: $head_node ($head_node_ip)"

# Set seeds
seeds=(42)  # Add more seeds if needed

model_name="Llama-3.1-70B"

lr=1.0e-6
exp_name="tulu3"

# settings
# ------------------------------------------------
# weight_type="original"
# dataset_name="tulu3_original"

# weight_type="equal"
# dataset_name="tulu3_equal"

# weight_type="submodular"
# dataset_name="tulu3_submodular"

weight_type="ours"
dataset_name="tulu3_ours"
# ------------------------------------------------

for seed in "${seeds[@]}"; do
    echo "Running with seed ${seed}"
    output_dir="saves/data_mixing/${exp_name}/${model_name}/${weight_type}/${model_name}_${dataset_name}_${lr}_seed${seed}"

    srun torchrun \
        --nproc_per_node=8 \
        --nnodes=16 \
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
        --output_dir ${output_dir} \
        --logging_steps 50 \
        --save_steps 50 \
        --plot_loss \
        --overwrite_output_dir \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 2 \
        --learning_rate ${lr} \
        --dispatch_batches False \
        --lr_scheduler_type linear \
        --warmup_ratio 0.05 \
        --bf16 \
        --ddp_timeout 180000 \
        --per_device_eval_batch_size 1 \
        --eval_strategy steps \
        --eval_dataset tulu3_general_val,tulu3_knowledge_recall_val,tulu3_math_val,tulu3_code_val,tulu3_safety_val,tulu3_precise_IF_val \
        --eval_steps 50 \
        --load_best_model_at_end True \
        --save_total_limit 1 \
        --include_num_input_tokens_seen True \
        --seed "${seed}" \
        --streaming True \
        --max_steps 4500 \
        >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
        2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"
done
