#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "data_mixing"
DATASET_INFO_PATH = DATA_DIR / "dataset_info.json"


DOMAIN_DATASETS: Dict[str, Path] = {
    "insfo": DATA_DIR / "tulu-3-sft-personas-instruction-following.json",
    "math": DATA_DIR / "tulu-3-sft-personas-math-filtered.json",
    "code": DATA_DIR / "tulu-3-sft-personas-code.json",
    "algebra": DATA_DIR / "tulu-3-sft-personas-algebra.json",
    "math-grade": DATA_DIR / "tulu-3-sft-personas-math-grade-filtered.json",
}

CUSTOM_RATIOS = {
    "insfo": 0.38268149,
    "math": 0.12392484,
    "code": 0.18790563,
    "algebra": 0.213406,
    "math-grade": 0.09208204
}

# 基础的 token 数量
BASE_TOKEN = 200_000_000


def sample_tokens(data: List[Dict], token_limit: int) -> List[Dict]:
    selected: List[Dict] = []
    total_tokens = 0
    while total_tokens <= token_limit:
        for item in data:
            if total_tokens + item.get("len", 0) <= token_limit:
                total_tokens += item.get("len", 0)
                selected.append({k: v for k, v in item.items() if k != "len"})
            else:
                total_tokens += item.get("len", 0)
                remaining_tokens = max(token_limit - total_tokens, 0)
                truncated_item = {k: v for k, v in item.items() if k != "len"}
                if remaining_tokens > 0 and isinstance(truncated_item.get("output"), str):
                    truncated_item["output"] = truncated_item["output"][:remaining_tokens]
                selected.append(truncated_item)
                break
        print(f"selected {total_tokens} tokens with limit {token_limit}")
        # 如果数据不够一轮，跳出避免死循环。如果支持重复采样，可以继续循环。
        break
    return selected


def load_json(path: Path) -> List[Dict]:
    with path.open("r") as f:
        return json.load(f)


def save_json(path: Path, payload: Iterable[Dict]) -> None:
    with path.open("w") as f:
        json.dump(list(payload), f, indent=2)


def ensure_dataset_info() -> Dict[str, Dict[str, str]]:
    if DATASET_INFO_PATH.exists():
        with DATASET_INFO_PATH.open("r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def persist_dataset_info(dataset_info: Dict[str, Dict[str, str]]) -> None:
    with DATASET_INFO_PATH.open("w") as f:
        json.dump(dataset_info, f, indent=2)


def run_custom_sampling(base_token: int = BASE_TOKEN) -> None:
    experiment_name = "custom_ratio_sampling"
    experiment_dir = OUTPUT_DIR / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = ensure_dataset_info()

    combined_dataset: List[Dict] = []
    
    for domain, ratio in CUSTOM_RATIOS.items():
        if domain not in DOMAIN_DATASETS:
            print(f"Warning: Dataset path for domain {domain} not found.")
            continue
            
        dataset_path = DOMAIN_DATASETS[domain]
        if not dataset_path.exists():
            print(f"Error: {dataset_path} does not exist.")
            continue
            
        structured_data = load_json(dataset_path)
        random.shuffle(structured_data)

        limit = int(base_token * ratio)
        print(f"\nProcessing domain: {domain}, ratio: {ratio}, target tokens: {limit}")

        sampled_items = sample_tokens(structured_data, limit)
        
        # 保存单一领域的采样结果
        subset_name = f"{base_token}_{domain}_custom.json"
        subset_path = experiment_dir / subset_name
        save_json(subset_path, sampled_items)
        print(f"Saved {subset_path.name} with {len(sampled_items)} items")

        relative_subset = Path("data_mixing") / experiment_name / subset_name
        dataset_key = f"{base_token}_{domain}_custom"
        dataset_info[dataset_key] = {"file_name": str(relative_subset)}
        
        combined_dataset.extend(sampled_items)

    # 混合所有领域的数据并保存
    random.shuffle(combined_dataset)
    combined_name = f"{base_token}_custom_mixed.json"
    combined_path = experiment_dir / combined_name
    save_json(combined_path, combined_dataset)
    print(f"\nSaved combined dataset {combined_path.name} with {len(combined_dataset)} items")

    relative_combined = Path("data_mixing") / experiment_name / combined_name
    dataset_key = f"{base_token}_custom_mixed"
    dataset_info[dataset_key] = {"file_name": str(relative_combined)}

    persist_dataset_info(dataset_info)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample data mixes with specific ratios.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional random seed for reproducibility."
    )
    parser.add_argument(
        "--base_token",
        type=int,
        default=2_000_000,
        help="Total token limit for the final combined dataset."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    
    run_custom_sampling(base_token=args.base_token)


if __name__ == "__main__":
    main()
