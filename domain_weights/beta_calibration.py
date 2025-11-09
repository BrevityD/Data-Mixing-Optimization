import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from scipy.optimize import Bounds, BFGS, minimize


# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------
DATASETS = {
    "llama-3.2-3b": {
        "base_token": 0.66,
        "n_multiplier": 3.0,
        "lyst": [
            [2.309398127536653, 2.2919550197611414, 2.2601072040502985, 2.2253171055400482, 2.2078428253445472],
            [2.2891207768582373, 2.275715067349909, 2.2601072040502985, 2.2340252519100123, 2.2223123883020706],
            [2.2946755321189904, 2.2822197449260098, 2.2601072040502985, 2.2327336034851544, 2.2183251314478096],
        ],
        "beta_estimates": [0.0510, 0.0430, 0.0439],
        "a_upper_ratio": 2.0 / 3.0,
        "inner_ratio": 2.0 / 3.0,
        "init_guess": [0.5, 0.5],
        "delta": 0.001,
        "use_log_ratio": True,
        "use_huber": True,
    },
    "orca": {
        "base_token": 0.66,
        "n_multiplier": 4.0,
        "lyst": [
            [2.024321116482459, 2.0105952584479216, 1.99, 1.9849282346653605, 1.9753737257904833],
            [2.0217433752861744, 2.0135445270790187, 1.99, 1.9759611105003072, 1.9533850007715063],
            [2.003216376124854, 2.004087887878923, 1.99, 1.9893702644617437, 1.9670705488252318],
            [2.0286690995573027, 2.0050834446692827, 1.99, 1.9858337752939834, 1.97776485902538],
        ],
        "beta_estimates": [0.0663, 0.0391, 0.0583, 0.0907],
        "a_upper_ratio": 3.0 / 4.0,
        "inner_ratio": 3.0 / 4.0,
        "init_guess": [0.2, 0.8],
        "delta": 0.01,
        "use_log_ratio": True,
        "use_huber": True,
    },
}


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------
def difference_model(Ni: float, Nj: float, beta: float, A: float) -> float:
    return (Ni + A) ** (-beta) - (Nj + A) ** (-beta)


def ratio_model(Ni: float, Nj: float, Nk: float, Nm: float, beta: float, A: float) -> float:
    numerator = difference_model(Ni, Nj, beta, A)
    denominator = difference_model(Nk, Nm, beta, A)
    return numerator / denominator


def huber_loss(residual: float, delta: float = 0.001) -> float:
    abs_r = abs(residual)
    if abs_r <= delta:
        return 0.5 * residual**2
    return delta * (abs_r - 0.5 * delta)


def multi_ratio_objective(ba, Lvals, Nvals, ratio_pairs, use_log_ratio, use_huber, delta):
    beta, A = ba
    total_loss = 0.0

    for (i, j), (k, m) in ratio_pairs:
        i0, j0 = i - 1, j - 1
        k0, m0 = k - 1, m - 1

        obs_num = Lvals[i0] - Lvals[j0]
        obs_den = Lvals[k0] - Lvals[m0]
        if abs(obs_den) < 1e-12:
            continue

        mod_num = difference_model(Nvals[i0], Nvals[j0], beta, A)
        mod_den = difference_model(Nvals[k0], Nvals[m0], beta, A)
        if abs(mod_den) < 1e-8:
            continue

        if use_log_ratio:
            if obs_num <= 0 or obs_den <= 0 or mod_num <= 0 or mod_den <= 0:
                continue
            obs_ratio = np.log(obs_num) / np.log(obs_den)
            mod_ratio = np.log(mod_num) / np.log(mod_den)
        else:
            obs_ratio = obs_num / obs_den
            mod_ratio = mod_num / mod_den

        diff = obs_ratio - mod_ratio
        if use_huber:
            total_loss += huber_loss(diff, delta=delta)
        else:
            total_loss += diff**2

    return total_loss


def generate_all_ratio_pairs():
    base_pairs = [
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 5),
        (3, 5),
    ]
    return [(num, den) for num, den in permutations(base_pairs, 2) if num != den]


def estimate_beta_and_A(Lvals, Nvals, config):
    ratio_pairs = generate_all_ratio_pairs()

    def objective(ba):
        return multi_ratio_objective(
            ba,
            Lvals,
            Nvals,
            ratio_pairs,
            config["use_log_ratio"],
            config["use_huber"],
            config["delta"],
        )

    bounds = Bounds(
        [0.0, 0.0],
        [np.inf, config["a_upper_ratio"] * config["n_multiplier"] * config["base_token"]],
    )

    result = minimize(
        objective,
        x0=config["init_guess"],
        method="trust-constr",
        bounds=bounds,
        jac="2-point",
        hess=BFGS(),
        options={"verbose": 0, "maxiter": 1000},
    )

    if result.success:
        beta, A = result.x
        return beta, A, result.fun, ratio_pairs
    return None, None, None, ratio_pairs


def plot_ratios(domain_index, beta, A, Lvals, Nvals, ratio_pairs):
    obs_list = []
    mod_list = []

    for (i, j), (k, m) in ratio_pairs:
        i0, j0 = i - 1, j - 1
        k0, m0 = k - 1, m - 1

        obs_num = Lvals[i0] - Lvals[j0]
        obs_den = Lvals[k0] - Lvals[m0]
        if abs(obs_den) < 1e-8:
            continue

        mod_num = difference_model(Nvals[i0], Nvals[j0], beta, A)
        mod_den = difference_model(Nvals[k0], Nvals[m0], beta, A)
        if abs(mod_den) < 1e-8:
            continue

        obs_list.append(obs_num / obs_den)
        mod_list.append(mod_num / mod_den)

    indices = np.arange(len(obs_list))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(indices - width / 2, obs_list, width=width, label="Observed")
    plt.bar(indices + width / 2, mod_list, width=width, label="Modeled")

    plt.title(f"Domain {domain_index} — Beta={beta:.4f}, A={A:.4f}")
    plt.xlabel("Ratio Pair Index")
    plt.ylabel("Ratio Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_dataset(name: str, config: dict) -> None:
    base_token = config["base_token"]
    N = config["n_multiplier"] * base_token
    N_vals = np.array([base_token * factor for factor in (1 / 3, 1 / 2, 1, 2, 3)])

    print(f"\n=== Running beta calibration for {name} ===")
    for domain_index, Lvals_raw in enumerate(config["lyst"], start=1):
        Lvals = np.array(Lvals_raw)
        beta_est, A_est, loss, ratio_pairs = estimate_beta_and_A(Lvals, N_vals, config)

        if beta_est is None:
            print(f"[Domain {domain_index}] Optimization failed.")
            continue

        print(
            f"[Domain {domain_index}] Beta={beta_est:.4f}, "
            f"A={A_est:.4f}, Loss={loss:.6f}"
        )
        plot_ratios(domain_index, beta_est, A_est, Lvals, N_vals, ratio_pairs)


if __name__ == "__main__":
    SELECTED_DATASETS = ["llama-3.2-3b", "orca"]
    for dataset_name in SELECTED_DATASETS:
        run_dataset(dataset_name, DATASETS[dataset_name])

