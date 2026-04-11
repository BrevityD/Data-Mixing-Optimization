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
    "llama-3.2-3b-mine": [
        {'beta': 0.051, 'c': 1.1684086283497737, 'gamma': 0.2028506624087608, 'alpha': 0.4835430572187707, 'e': 1.0850955259279713, 'loss': 5.27289724525855e-06},
        {'beta': 0.043, 'c': 1.0400851550373114, 'gamma': 0.30771406857207306, 'alpha': 0.48615169878903874, 'e': 1.21897530701965, 'loss': 3.1662223580762807e-06},
        {'beta': 0.0439, 'c': 1.1943096075416968, 'gamma': 0.31982181823990985, 'alpha': 0.48608269597594067, 'e': 1.0672524326328172, 'loss': 2.616888091466727e-06},
    ],
    "orca-mine": [
        {'beta': 0.0663, 'c': 0.2862807935222693, 'gamma': 0.002847823167745079, 'alpha': 0.2393850912392418, 'e': 1.7022100069734682, 'loss': 7.293530876850732e-06},
        {'beta': 0.0583, 'c': 1.000064063739959, 'gamma': 1.1967188479138744, 'alpha': 0.586790339867987, 'e': 1.0419043858508832, 'loss': 1.6144187881281025e-05},
        {'beta': 0.0907, 'c': 0.21092234330219178, 'gamma': 0.00012156143964921182, 'alpha': 0.3842510890342751, 'e': 1.7793320126218384, 'loss': 1.5672137841258513e-05},
    ],
    "reprod": [
        {'beta': 0.0025, 'c': 0.6438822970090471, 'gamma': 0.04298530162957, 'alpha': 0.2977915700418385, 'e': 0.8415645944959887, 'loss': 2.610253979330633e-08},
        {'beta': 0.0066, 'c': 1.0049334248113198, 'gamma': 0.07784550996776536, 'alpha': 0.23974814961733348, 'e': 0.484284137477415, 'loss': 6.504852319686068e-09},
        {'beta': 0.0302, 'c': 0.11491131047305071, 'gamma': 0.002084546537675324, 'alpha': 0.28382230031041555, 'e': 1.3723727842828402, 'loss': 1.5935130213300665e-06},
        {'beta': 0.004, 'c': 0.9441718507018281, 'gamma': 0.38347775747630475, 'alpha': 0.4905550013695846, 'e': 0.5444781808424246, 'loss': 4.4661569559995846e-08},
        {'beta': 0.0018, 'c': 1.163021071515249, 'gamma': 0.18555491737003274, 'alpha': 0.31375376650764425, 'e': 0.3229289039292824, 'loss': 8.585062590385621e-08},
    ]
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
        # plot_weights(N_VALUES, weights, title=f"Optimal Weights — {name}")

