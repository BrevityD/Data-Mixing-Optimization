""" 随便训训
"""

import argparse
import subprocess
from pathlib import Path
from loguru import logger

def main():
    parser = argparse.ArgumentParser(
        description="LLaMA-Factory"
    )
    parser.add_argument(
        "--template_yaml", type=str, required=True,
        help="Path to the base YAML config for LLaMA-Factory"
    )
    parser.add_argument(
        "--base_token", type=int, default=200000000,
        help="Base token amount used in dataset names (default: 200000000)"
    )
    parser.add_argument(
        "--model_name", type=str, default="Llama-3.1-8B",
        help="Model name for output directory naming (default: Llama-3.1-8B)"
    )
    parser.add_argument(
        "--output_root", type=str, default="saves/final",
        help="Root directory for saving checkpoints (default: saves/exp2_grid)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing"
    )
    args = parser.parse_args()
    base_token = args.base_token

    dataset_str = f"{base_token}_custom_mixed"

    # 输出目录命名，保留比例信息
    output_dir = (
        Path(args.output_root) /
        f"{args.model_name}_{base_token}" /
        dataset_str
    )

    cmd = [
        "llamafactory-cli", "train", args.template_yaml,
        f"dataset={dataset_str}",
        f"output_dir={str(output_dir)}",
        f"run_name={args.model_name}_final_{base_token}"
    ]

    logger.info(" ".join(cmd))

    if not args.dry_run:
        # 确保输出目录存在（llamafactory 也会自动创建，但提前建一下无妨）
        output_dir.mkdir(parents=True, exist_ok=True)
        # 执行训练
        result = subprocess.run(cmd)

    logger.info("\nAll training jobs finished.")


if __name__ == "__main__":
    main()
