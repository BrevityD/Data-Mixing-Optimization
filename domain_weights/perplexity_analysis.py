import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def default_results_root():
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "results" / "data_mixing"


def collect_average_perplexity(root_dir, model_name, token_bucket, domains, sizes, prefix=""):
    root = Path(root_dir) / model_name / token_bucket
    records = []

    for domain in domains:
        for size in sizes:
            acc = 0.0
            success = 0

            for val_domain in domains:
                other_domains = [d for d in domains if d != domain]
                filename = (
                    f"ppl_{prefix}{val_domain}_val_{model_name}_{domain}_{size}_"
                    f"{other_domains[0]}_one_{other_domains[1]}_one.json"
                )
                filepath = root / domain / filename
                if not filepath.is_file():
                    print(f"Warning: missing file {filepath}")
                    continue
                try:
                    data = json.loads(filepath.read_text())
                    token_ppl = data[0]["token-level ppl"]
                except (json.JSONDecodeError, IndexError, KeyError, TypeError) as err:
                    print(f"Warning: failed to parse {filepath}: {err}")
                    continue

                acc += token_ppl
                success += 1

            avg = acc / success if success else None
            records.append(
                {
                    "train_domain": domain,
                    "size": size,
                    "avg_ppl": avg,
                    "count": success,
                }
            )
    return records


def collect_full_matrix(root_dir, model_name, token_bucket, domains, sizes, prefix=""):
    root = Path(root_dir) / model_name / token_bucket
    rows = []

    for domain in domains:
        for size in sizes:
            for val_domain in domains:
                other_domains = [d for d in domains if d != domain]
                filename = (
                    f"ppl_{prefix}{val_domain}_val_{model_name}_{domain}_{size}_"
                    f"{other_domains[0]}_one_{other_domains[1]}_one.json"
                )
                filepath = root / domain / filename
                if not filepath.is_file():
                    continue
                try:
                    data = json.loads(filepath.read_text())
                    token_ppl = data[0]["token-level ppl"]
                except (json.JSONDecodeError, IndexError, KeyError, TypeError):
                    continue
                rows.append(
                    {
                        "train_domain": domain,
                        "val_domain": val_domain,
                        "size": size,
                        "token_ppl": token_ppl,
                    }
                )
    return pd.DataFrame(rows)


def create_custom_blue_cmap():
    colors = ["#163d71", "#5e6983", "#e8ecf2"]
    return LinearSegmentedColormap.from_list("custom_blue", colors, N=256)


def plot_heatmap(matrix, title, cmap, save_path=None):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6), dpi=300)
    heatmap = sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 16, "weight": "bold"},
        square=True,
        cbar_kws={"shrink": 0.75, "aspect": 20},
    )
    heatmap.collections[0].colorbar.ax.tick_params(labelsize=14)
    plt.xlabel("Validation Domain", fontsize=16)
    plt.ylabel("Training Domain", fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.title(title, fontsize=18)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def build_difference_matrix(df, size_a, size_b, domains):
    pivot_a = df[df["size"] == size_a].pivot(index="train_domain", columns="val_domain", values="token_ppl")
    pivot_b = df[df["size"] == size_b].pivot(index="train_domain", columns="val_domain", values="token_ppl")
    pivot_a = pivot_a.reindex(index=domains, columns=domains)
    pivot_b = pivot_b.reindex(index=domains, columns=domains)
    return pivot_b - pivot_a


if __name__ == "__main__":
    ROOT = default_results_root()
    DOMAINS = ["instr", "math", "code"]
    SIZES = ["third", "half", "one", "double", "triple"]

    CONFIGS = {
        "llama-3.2-3b": {"token_bucket": "660000", "prefix": ""},
        "llama-3.1-8b": {"token_bucket": "660000", "prefix": ""},
        "llama-3.1-8b-alt": {"token_bucket": "660000", "prefix": "5000000_"},
    }

    for model_name, cfg in CONFIGS.items():
        print(f"\n=== Processing {model_name} ===")
        df = collect_full_matrix(ROOT, model_name, cfg["token_bucket"], DOMAINS, SIZES, prefix=cfg["prefix"])
        if df.empty:
            print("No data found.")
            continue

        for size in ("one", "triple"):
            matrix = df[df["size"] == size].pivot(
                index="train_domain", columns="val_domain", values="token_ppl"
            ).reindex(index=DOMAINS, columns=DOMAINS)
            if matrix.isnull().all().all():
                continue
            plot_heatmap(
                matrix,
                title=f"{model_name} — size={size}",
                cmap=create_custom_blue_cmap(),
                save_path=Path("heatmaps") / f"{model_name}_{size}.pdf",
            )

        diff_matrix = build_difference_matrix(df, "one", "triple", DOMAINS)
        if not diff_matrix.isnull().all().all():
            plot_heatmap(
                diff_matrix,
                title=f"{model_name} — triple minus one",
                cmap=create_custom_blue_cmap(),
                save_path=Path("heatmaps") / f"{model_name}_difference.pdf",
            )

