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
    "code": DATA_DIR / "tulu-3-sft-personas-code.json",
    "insfo": DATA_DIR / "tulu-3-sft-personas-instruction-following.json",
    "math": DATA_DIR / "tulu-3-sft-personas-math-filtered.json",
    "algebra": DATA_DIR / "tulu-3-sft-personas-algebra.json",
    "math-grade": DATA_DIR / "tulu-3-sft-personas-math-grade-filtered.json",
}


EXPERIMENTS: Dict[str, Dict] = {
    "calibrator": {
        "experiment_name": "calibrator",
        "base_token": 2_000_000,
        "token_limit_ratios": {
            "0.25": 0.25,
            "0.5": 0.5,
            "1": 1,
            "2": 2,
            "3": 3,
        },
        "match_domain_in_key": False,
        "validation_size": 1_000,
        "combine_validation": True,
    },
    "exp2": {
        "experiment_name": "exp2",
        "base_token": 2_000_000,
        "token_limit_ratios": {
            "0.125": 0.125,
            "0.25": 0.25,
            "0.375": 0.375,
            "0.5": 0.5,
            "0.625": 0.625,
            "0.75": 0.75,
        },
        "match_domain_in_key": False,
        "validation_size": 1_000,
        "combine_validation": True,
    }
}


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


def pick_validation(
    structured_data: List[Dict], validation_size: int
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    validation_set = random.sample(structured_data, validation_size)
    validation_ids = {json.dumps(item, sort_keys=True) for item in validation_set}
    filtered_validation = [{k: v for k, v in item.items() if k != "len"} for item in validation_set]
    remaining = [item for item in structured_data if json.dumps(item, sort_keys=True) not in validation_ids]
    random.shuffle(remaining)
    return validation_set, filtered_validation, remaining


def resolve_token_limits(config: Dict, base_token: int) -> Dict[str, int]:
    if "token_limits" in config:
        return config["token_limits"].copy()
    ratios = config.get("token_limit_ratios", {})
    return {label: int(base_token * ratio) for label, ratio in ratios.items()}


def run_experiment(experiment_key: str) -> None:
    if experiment_key not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_key}")

    config = EXPERIMENTS[experiment_key]
    base_token = config["base_token"]
    experiment_name = config["experiment_name"]
    validation_size: Optional[int] = config.get("validation_size")
    combine_validation = bool(config.get("combine_validation"))
    match_domain = bool(config.get("match_domain_in_key"))

    token_limits = resolve_token_limits(config, base_token)

    experiment_dir = OUTPUT_DIR / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = ensure_dataset_info()
    combined_validation: List[Dict] = []

    for domain, dataset_path in DOMAIN_DATASETS.items():
        structured_data = load_json(dataset_path)

        filtered_data = structured_data
        if validation_size:
            _, filtered_validation, filtered_data = pick_validation(structured_data, validation_size)
            val_filename = f"{base_token}_{domain}_val.json"
            val_path = experiment_dir / val_filename
            save_json(val_path, filtered_validation)
            print(f"Saved {val_path.name} with {len(filtered_validation)} items")

            relative_path = Path("data_mixing") / experiment_name / val_filename
            dataset_info[f"{base_token}_{domain}_val"] = {"file_name": str(relative_path)}

            if combine_validation:
                combined_validation.extend(filtered_validation)

        random.shuffle(filtered_data)

        for label, limit in token_limits.items():
            if match_domain and domain not in label:
                continue

            sampled_items = sample_tokens(filtered_data, limit)
            subset_name = f"{base_token}_{domain}_{label}.json"
            subset_path = experiment_dir / subset_name
            save_json(subset_path, sampled_items)
            print(f"Saved {subset_path.name} with {len(sampled_items)} items")

            relative_subset = Path("data_mixing") / experiment_name / subset_name
            dataset_key = f"{base_token}_{domain}_{label}"
            dataset_info[dataset_key] = {"file_name": str(relative_subset)}

    if combine_validation and combined_validation:
        combined_name = f"{base_token}_{experiment_name}_val.json"
        combined_path = experiment_dir / combined_name
        save_json(combined_path, combined_validation)
        print(f"Saved {combined_path.name} with {len(combined_validation)} items")

        relative_combined = Path("data_mixing") / experiment_name / combined_name
        dataset_key = f"{base_token}_{experiment_name}_val"
        dataset_info[dataset_key] = {"file_name": str(relative_combined)}

    persist_dataset_info(dataset_info)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample data mixes for designated experiments.")
    parser.add_argument(
        "experiment",
        choices=sorted(EXPERIMENTS.keys()),
        help="Experiment key to run."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    run_experiment(args.experiment)


if __name__ == "__main__":
    main()
