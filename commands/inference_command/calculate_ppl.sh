#!/bin/bash
#SBATCH --partition=mbzuai
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive

module load cuda/12.4
source ~/miniconda3/bin/activate myenv
cd /mbz/users/liyuan/LLaMA-Factory
# Set NCCL to debug only errors
export NCCL_DEBUG=WARN

# Disable NVML if not required
export PYTORCH_NO_NVML=1

# Add recommended NCCL settings
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3

# Navigate to your project directory
lr=1.0e-5

current_time=$(date +"%Y%m%d_%H%M%S")

checkpoint_path="/mbz/users/liyuan/LLaMA-Factory/saves/data_mixing/exp1"
# model_base="Llama-3.2-3B"
model_base="Llama-3.1-8B"
base_token="660000"
experiment_result="/mbz/users/liyuan/LLaMA-Factory/results/data_mixing/exp1"
domains=("instr" "math" "code")

sizes=("one" "half" "third" "double" "triple")
# sizes=("onehalf")

# eval_dataset="5000000_exp2_val"
eval_datasets=("5000000_math_val" "5000000_code_val" "5000000_instr_val")

for domain in "${domains[@]}"; do
    for size in "${sizes[@]}"; do
        echo "Processing domain: $domain, size: $size"

        dataset1="${domain}_${size}"

        # Dataset 2 and 3 should be from the other domains with the size "one"
        other_domains=()
        for d in "${domains[@]}"; do
            if [[ "$d" != "$domain" ]]; then
                other_domains+=("$d")
            fi
        done

        echo "Processing other domains: ${other_domains[@]}"
        dataset2="${other_domains[0]}_one"
        dataset3="${other_domains[1]}_one"

        mkdir -p ${experiment_result}/${model_base}/${base_token}/${domain}

        for eval_dataset in "${eval_datasets[@]}"; do 
            torchrun /mbz/users/liyuan/LLaMA-Factory/scripts/cal_ppl.py \
            --model_name_or_path ${checkpoint_path}/${model_base}/${base_token}_4/${domain}/${model_base}_${dataset1}_${dataset2}_${dataset3}_${lr}_seed42 \
            --save_name ${experiment_result}/${model_base}/${base_token}/${domain}/ppl_${eval_dataset}_${model_base}_${dataset1}_${dataset2}_${dataset3}.json \
            --batch_size 2 \
            --dataset ${eval_dataset} \
            --template llama3 \
            --cutoff_len 4096 \
            >> "printout/output_file/output_inference_${SLURM_JOB_ID}.out" \
                2>> "printout/error_file/error_inference_${SLURM_JOB_ID}.err"
        done
    done
done


# size_pairs=(
#     "instr triple"
#     "instr third"
#     "math triple"
#     "code triple"
# )

# lr=1.0e-5

# for pair in "${size_pairs[@]}"; do
#     read -r domain size <<< "$pair"

#     echo "Processing domain: $domain, size: $size"

#     dataset1="${domain}_${size}"

#     # Dataset 2 and 3 should be from the other domains with the size "one"
#     other_domains=()
#     for d in "${domains[@]}"; do
#         if [[ "$d" != "$domain" ]]; then
#             other_domains+=("$d")
#         fi
#     done

#     echo "Processing other domains: ${other_domains[@]}"
#     dataset2="${other_domains[0]}_one"
#     dataset3="${other_domains[1]}_one"

#     mkdir -p ${experiment_result}/${model_base}/${base_token}_2/${domain}

#     for eval_dataset in "${eval_datasets[@]}"; do 
#         torchrun /mbz/users/liyuan/LLaMA-Factory/scripts/cal_ppl.py \
#         --model_name_or_path ${checkpoint_path}/${model_base}/${base_token}_2/${domain}/${model_base}_${dataset1}_${dataset2}_${dataset3}_${lr}_seed42 \
#         --save_name ${experiment_result}/${model_base}/${base_token}_2/${domain}/ppl_${eval_dataset}_${model_base}_${dataset1}_${dataset2}_${dataset3}.json \
#         --batch_size 2 \
#         --dataset ${eval_dataset} \
#         --template llama3 \
#         --cutoff_len 4096 \
#         >> "printout/output_file/output_inference_${SLURM_JOB_ID}.out" \
#             2>> "printout/error_file/error_inference_${SLURM_JOB_ID}.err"
#     done
# done