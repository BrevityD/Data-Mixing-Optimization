import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.optimize import minimize


PARAM_SETS = {
    "llama-3.2-3b": [
        {"beta": 0.051, "c": 1.156484125642086, "gamma": 0.19780651007960018, "alpha": 0.4759641653939281, "e": 1.0964457298437293},
        {"beta": 0.043, "c": 0.7508495582040167, "gamma": 0.040025262958907154, "alpha": 0.43503938591132635, "e": 1.4937431374616452},
        {"beta": 0.0439, "c": 0.9828179869932256, "gamma": 0.1260596510875358, "alpha": 0.46702823082847034, "e": 1.267061465923874},
    ],
    "orca": [
        {"beta": 0.1074, "c": 0.9303720966305796, "gamma": 0.8353839985461634, "alpha": 0.5601292648858495, "e": 1.4226529933436454},
        {"beta": 0.2056, "c": 0.47001708560647254, "gamma": 1.1442480438863765, "alpha": 0.5074276704841271, "e": 1.8919117116032493},
        {"beta": 0.2791, "c": 0.36073465526930765, "gamma": 1.106071287791263, "alpha": 0.5113088104398635, "e": 1.9937865053832902},
    ],
}

DOMAIN_NAMES = ["IF", "Math", "Code"]
DOMAIN_COLORS = ["#344F99", "#8CB1CA", "#989898"]


def objective(w, N, params):
    total = 0.0
    for i, p in enumerate(params):
        expr = w[i] * N + p["gamma"] * (N - w[i] * N) ** p["alpha"]
        total += p["c"] * (expr ** (-p["beta"])) + p["e"]
    return total


def optimize_weights(N, params):
    k = len(params)
    initial = np.full(k, 1.0 / k)
    bounds = [(0.0, 1.0)] * k
    constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    result = minimize(
        fun=objective,
        x0=initial,
        args=(N, params),
        method="SLSQP",
        bounds=bounds,
        constraints=[constraint],
        options={"disp": False},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed for N={N}: {result.message}")

    return result.x, result.fun


def sweep_values(N_values, params):
    weights = []
    objectives = []
    for N in N_values:
        w_opt, obj = optimize_weights(N, params)
        weights.append(w_opt)
        objectives.append(obj)
    return np.array(weights), np.array(objectives)


def get_color(index):
    return DOMAIN_COLORS[index % len(DOMAIN_COLORS)]


def plot_weights(N_values, weights, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx in range(weights.shape[1]):
        ax.plot(
            N_values,
            weights[:, idx],
            marker="o",
            color=get_color(idx),
            linestyle="-",
            linewidth=4,
            markersize=10,
            label=DOMAIN_NAMES[idx],
        )

    ax.set_xlabel(r"$N_0$ (1e6 tokens)", fontsize=20)
    ax.set_ylabel("Estimated Weight", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, linestyle="--", linewidth=1, alpha=0.7)
    ax.legend(fontsize=18)
    ax.set_title(title, fontsize=20)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    N_VALUES = [5, 10, 20, 50, 100, 200, 500]
    for name, params in PARAM_SETS.items():
        print(f"\n=== Optimizing weights for {name} ===")
        weights, objectives = sweep_values(N_VALUES, params)
        for N, w, obj in zip(N_VALUES, weights, objectives):
            print(f"N={N:>4}: weights={w}, objective={obj:.6f}")
        plot_weights(N_VALUES, weights, title=f"Optimal Weights — {name}")

