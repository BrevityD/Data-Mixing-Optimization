#!/bin/bash
#SBATCH --partition=highprio
#SBATCH --time=50:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --job-name=lm_eval
#SBATCH --output=printout/job_%j.out
#SBATCH --error=printout/job_%j.err

module load cuda/12.1
source ~/miniconda3/bin/activate myenv

cd /mbz/users/liyuan/LLaMA-Factory

groups

docker run hello-world