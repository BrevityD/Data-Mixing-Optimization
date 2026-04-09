#!/usr/bin/env python3
"""
"""

import argparse
import itertools
import subprocess
from decimal import Decimal, getcontext
from pathlib import Path
from loguru import logger

# 设置高精度比较，避免浮点数误差
getcontext().prec = 6


def check_sum_one(*ratios: str) -> bool:
    """检查比例字符串之和是否为 1"""
    total = sum(Decimal(r) for r in ratios)
    return abs(total - Decimal("1")) < Decimal("1e-6")


def generate_combinations(domains, sizes):
    """生成所有比例和为 1 的组合，返回列表，每项为 (domain1_ratio, domain2_ratio, domain3_ratio)"""
    valid = []
    for s1, s2, s3, s4, s5 in itertools.product(sizes, repeat=5):
        if check_sum_one(s1, s2, s3, s4, s5):
            valid.append((s1, s2, s3, s4, s5))
    logger.info(f"will train {len(valid)} ratioes.")
    return valid


def main():
    parser = argparse.ArgumentParser(
        description="LLaMA-Factory grid search for exp2 data mixing"
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
        "--output_root", type=str, default="saves/exp2_grid",
        help="Root directory for saving checkpoints (default: saves/exp2_grid)"
    )
    parser.add_argument(
        "--domains", nargs="+", default=["insfo", "math", "code", "algebra", "math-grade"],
        help="List of domain names (default: )"
    )
    parser.add_argument(
        "--sizes", nargs="+", default=["0.125", "0.25", "0.375", "0.5", "0.625", "0.75"],
        help="List of ratio strings to grid over (default: 0.125 0.25 0.375 0.5 0.625 0.75)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing"
    )
    args = parser.parse_args()

    domains = args.domains
    sizes = args.sizes
    base_token = args.base_token

    combinations = generate_combinations(domains, sizes)

    for idx, (r1, r2, r3, r4, r5) in enumerate(combinations, 1):
        # 构造数据集名称，格式：{base_token}_{domain}_{ratio}
        dataset_str = (
            f"{base_token}_{domains[0]}_{r1},"
            f"{base_token}_{domains[1]}_{r2},"
            f"{base_token}_{domains[2]}_{r3},"
            f"{base_token}_{domains[3]}_{r4},"
            f"{base_token}_{domains[4]}_{r5}"
        )

        # 输出目录命名，保留比例信息
        output_dir = (
            Path(args.output_root) /
            f"{args.model_name}_{base_token}" /
            f"{domains[0]}{r1}_{domains[1]}{r2}_{domains[2]}{r3}_{domains[3]}{r4}_{domains[4]}{r5}"
        )

        cmd = [
            "llamafactory-cli", "train", args.template_yaml,
            f"dataset={dataset_str}",
            f"output_dir={str(output_dir)}",
            f"run_name={args.model_name}_{base_token}_{domains[0]}{r1}_{domains[1]}{r2}_{domains[2]}{r3}_{domains[3]}{r4}_{domains[4]}{r5}"
        ]

        logger.info(f"\n[{idx}/{len(combinations)}] Running:")
        logger.info(" ".join(cmd))

        if not args.dry_run:
            # 确保输出目录存在（llamafactory 也会自动创建，但提前建一下无妨）
            output_dir.mkdir(parents=True, exist_ok=True)
            # 执行训练
            result = subprocess.run(cmd)

    logger.info("\nAll training jobs finished.")


if __name__ == "__main__":
    main()