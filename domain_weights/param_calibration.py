import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import BFGS, Bounds, NonlinearConstraint, minimize

from .beta_calibration import DATASETS, huber_loss


def model(x_vals, c, gamma, alpha, beta, e, N, inner_ratio):
    inner_term = inner_ratio * N
    return c * (x_vals + gamma * (inner_term ** alpha)) ** (-beta) + e


def objective_fixed_beta(params, x_vals, y_vals, beta_fixed, N, inner_ratio, delta):
    log_c, log_gamma, alpha, log_e = params
    c_ = np.exp(log_c)
    gamma_ = np.exp(log_gamma)
    e_ = np.exp(log_e)

    y_pred = model(x_vals, c_, gamma_, alpha, beta_fixed, e_, N, inner_ratio)
    residuals = y_vals - y_pred
    return np.sum(huber_loss(residuals, delta=delta))


def constraint_fun(params, N, inner_ratio):
    _, log_gamma, alpha, _ = params
    gamma_ = np.exp(log_gamma)
    inner_term = inner_ratio * N
    return gamma_ * (inner_term ** alpha) - inner_term


def fit_domain_parameters(domain_index, Lvals_raw, beta_fixed, config, delta):
    base_token = config["base_token"]
    N = config["n_multiplier"] * base_token
    x_vals = np.array([base_token * v for v in (1 / 3, 1 / 2, 1, 2, 3)])
    y_vals = np.array(Lvals_raw)
    inner_ratio = config["inner_ratio"]

    def objective_fn(params, x, y):
        return objective_fixed_beta(params, x, y, beta_fixed, N, inner_ratio, delta)

    def constraint_fn(params):
        return constraint_fun(params, N, inner_ratio)

    constraint = NonlinearConstraint(
        fun=constraint_fn,
        lb=-np.inf,
        ub=0.0,
        jac="2-point",
        hess=BFGS(),
    )

    bounds = Bounds(
        lb=[-np.inf, -np.inf, 0.0, -np.inf],
        ub=[np.inf, np.inf, 1.0, np.inf],
    )

    initial_guess = [np.log(3.0), np.log(0.5), 0.5, np.log(1.0)]

    result = minimize(
        fun=objective_fn,
        x0=initial_guess,
        args=(x_vals, y_vals),
        method="trust-constr",
        jac="2-point",
        hess=BFGS(),
        bounds=bounds,
        constraints=[constraint],
        options={"maxiter": 5000, "verbose": 0},
    )

    if not result.success:
        print(f"[Domain {domain_index}] Optimization failed: {result.message}")
        return None

    log_c_opt, log_gamma_opt, alpha_opt, log_e_opt = result.x
    c_opt = np.exp(log_c_opt)
    gamma_opt = np.exp(log_gamma_opt)
    e_opt = np.exp(log_e_opt)

    fitted = {
        "beta": beta_fixed,
        "c": c_opt,
        "gamma": gamma_opt,
        "alpha": alpha_opt,
        "e": e_opt,
        "loss": result.fun,
    }

    x_fine = np.linspace(x_vals.min(), x_vals.max(), 200)
    y_fitted = model(x_fine, c_opt, gamma_opt, alpha_opt, beta_fixed, e_opt, N, inner_ratio)

    plt.figure()
    plt.scatter(x_vals, y_vals, color="red", label="Data")
    plt.plot(x_fine, y_fitted, color="blue", label="Fitted Model")
    plt.title(f"Domain {domain_index}: Beta = {beta_fixed:.4f} (fixed)")
    plt.xlabel("x")
    plt.ylabel("L(x)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    ratio = (gamma_opt * (inner_ratio * N) ** alpha_opt) / (inner_ratio * N)
    print(
        f"[Domain {domain_index}] c={c_opt:.6f}, gamma={gamma_opt:.6f}, "
        f"alpha={alpha_opt:.6f}, e={e_opt:.6f}, loss={result.fun:.6f}, "
        f"constraint_ratio={ratio:.6f}"
    )

    return fitted


def run_dataset(name: str, config: dict, delta: float):
    print(f"\n=== Parameter fitting for {name} ===")
    betas = config.get("beta_estimates")
    if not betas:
        print("  Skipping: beta estimates are missing.")
        return []

    fitted_params = []
    for domain_index, (beta_fixed, Lvals_raw) in enumerate(zip(betas, config["lyst"]), start=1):
        params = fit_domain_parameters(domain_index, Lvals_raw, beta_fixed, config, delta)
        if params:
            fitted_params.append(params)

    return fitted_params


if __name__ == "__main__":
    DELTA = 0.001
    results = {}
    for dataset_name in ["llama-3.2-3b", "orca"]:
        params = run_dataset(dataset_name, DATASETS[dataset_name], DELTA)
        results[dataset_name] = params

    print("\n--- Summary ---")
    for dataset_name, params in results.items():
        print(f"{dataset_name}:")
        for param in params:
            print(param)

