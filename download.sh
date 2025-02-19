#!/bin/bash
#SBATCH --partition=highprio
#SBATCH --time=10:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive

# cd /mbz/users/liyuan/LLaMA-Factory/checkpoints
# git clone https://huggingface.co/meta-llama/Llama-2-70b-hf

while true
do
    # Your code here
    echo "This loop will run forever."
    sleep 1  # Optional: Pause for 1 second to prevent flooding the terminal
done