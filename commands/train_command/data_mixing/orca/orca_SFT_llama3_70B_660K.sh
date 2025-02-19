#!/bin/bash
#SBATCH --partition=highprio
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
exp_name="orca"
base_token=660000
# domains=("t0" "cot" "flan" "niv")
domains=("niv")
# sizes=("one" "half" "third" "double" "triple")
sizes=("triple")

lr=1.0e-6

for domain in "${domains[@]}"; do
    for size in "${sizes[@]}"; do
        echo "Processing domain: $domain, size: $size"

        dataset1="${base_token}_${domain}_${size}"

        # Gather other domains excluding the current one
        other_domains=()
        for d in "${domains[@]}"; do
            if [[ "$d" != "$domain" ]]; then
                other_domains+=("$d")
            fi
        done

        echo "Processing other domains: ${other_domains[@]}"
        dataset2="${base_token}_${other_domains[0]}_one"
        dataset3="${base_token}_${other_domains[1]}_one"
        dataset4="${base_token}_${other_domains[2]}_one"

        # Debugging output
        echo "dataset1: $dataset1"
        echo "dataset2: $dataset2"
        echo "dataset3: $dataset3"
        echo "dataset4: $dataset4"
        echo "----------------------------------"

        # Loop over each seed
        for seed in "${seeds[@]}"; do
            echo "Running with seed ${seed}"

            output_dir="saves/data_mixing/${exp_name}/${model_name}/${base_token}/${domain}/${model_name}_${dataset1}_${dataset2}_${dataset3}_${dataset4}_${lr}_seed${seed}"

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
                --dataset ${dataset1},${dataset2},${dataset3},${dataset4} \
                --template llama3 \
                --cutoff_len 4096 \
                --overwrite_cache \
                --output_dir ${output_dir} \
                --logging_steps 5 \
                --save_steps 5 \
                --plot_loss \
                --overwrite_output_dir \
                --per_device_train_batch_size 1 \
                --gradient_accumulation_steps 2 \
                --learning_rate ${lr} \
                --dispatch_batches False \
                --lr_scheduler_type cosine \
                --warmup_ratio 0.05 \
                --bf16 \
                --ddp_timeout 180000 \
                --per_device_eval_batch_size 1 \
                --eval_strategy steps \
                --eval_dataset orca_t0_val,orca_cot_val,orca_flan_val,orca_niv_val \
                --eval_steps 5 \
                --load_best_model_at_end True \
                --save_total_limit 1 \
                --include_num_input_tokens_seen True \
                --seed "${seed}" \
                --streaming True \
                --max_steps 80 \
                >> "printout/output_file/output_${SLURM_JOB_ID}_${current_time}_${seed}.out" \
                2>> "printout/error_file/error_${SLURM_JOB_ID}_${current_time}_${seed}.err"
        done
    done
done
