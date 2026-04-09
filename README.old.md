# Data Mixing Optimization

This repository accompanies the paper [*Data Mixing Optimization for Supervised Fine-Tuning of Large Language Models*](https://arxiv.org/abs/2508.11953) by Yuan Li, Zhengzhong Liu, and Eric Xing.

## Environment Setup
- `conda create -n <venv_name> python=3.10` and `conda activate <venv_name>`.
- `pip install -r requirements.txt`.
- For evaluation jobs, the SLURM wrapper expects an environment named `lm-eval`; create it via `conda create -n lm-eval python=3.10` (or rename the env in the script to match your setup).
- Clone the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) repository and install it inside that environment, e.g. `git clone https://github.com/EleutherAI/lm-evaluation-harness && cd lm-evaluation-harness && pip install -e .`, so the `lm_eval` CLI invoked by `commands/eval_command/lm-evaluation-harness.sh` is available.
- `source commands/train_command/common_env.sh` to load shared CUDA, NCCL, and logging settings (edit defaults in the script as needed).

## Data Processing
- Each dataset-specific loader sits in `data_preprocess/<dataset_name>/` (e.g., `openorca`, `tulu3-sft`, `infinity_instruct`, `opencoder`, `openmathinstruct2`). Run the matching `load_data.py` (or Slurm wrapper) to download, filter, and tokenize that dataset, which writes cleaned JSON into `data/`.
- `data_preprocess/data_mixing/sample_data.py` combines those JSON files into domain mixtures, creates optional validation splits, and registers paths in `data/dataset_info.json`.
- All trainable model weights live under `checkpoints/`; populate it with the base models (e.g., `checkpoints/Llama-3.1-8B`) before launching jobs, matching the paths the scripts expect.
- Any dataset you plan to reference in launch scripts must have an entry in `data/dataset_info.json` pointing to its JSON file; the preprocessing scripts above append these entries automatically.

## Training & Inference
- Submit fine-tuning jobs with `sbatch commands/train_command/train_single.sh` (or switch to `data_mixing.sh` / `recipe_exp1.sh`). These wrappers mirror [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory); consult upstream docs for detailed implementation.
- Launch inference/perplexity sweeps with `sbatch commands/inference_command/calculate_ppl.sh domain` (use `total` for global perplexity runs). Adjust variables at the top of the script before submission.

## Evaluation
- `sbatch commands/eval_command/lm-evaluation-harness.sh` runs downstream tasks via the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness); edit the task/model arrays inside the script or follow harness documentation for custom task lists.

## Domain Weight Derivation
- Step 1: `python domain_weights/beta_calibration.py` (or run `domain_weights/standard_beta_cal.ipynb` via `jupyter lab domain_weights/`) to estimate beta.
- Step 2: `python domain_weights/param_calibration.py` or the companion notebook `domain_weights/param_cal.ipynb` to fit remaining parameters.
- Step 3: `python domain_weights/perplexity_analysis.py` to aggregate validation perplexities and visualize matrices; optionally `python domain_weights/weight_optimizer.py` to sweep final weights.
- Execute scripts/notebooks in the beta fit → parameter fit → perplexity order; notebooks contain worked examples mirroring the scripted flow.