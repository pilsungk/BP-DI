# ============================================================
# Step 1: Q_k scaling analysis
#
# Reconstruct:
#   Q_k = abs_sum_k^2 / N_eff_k
#
# Main estimator:
#   1. Mean over seeds for each parameter
#   2. Mean over parameters
#
# Uncertainty:
#   Bootstrap over seeds, treating seed as the independent
#   resampling unit.
#
# Scaling fit:
#   log(mean_Q) = slope * n + intercept
#   Report slope, per-qubit factor, R^2, and bootstrap CI.
# ============================================================

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

CSV_PATH = "../results/tfim/per_run.csv"

EXPECTED_SEEDS = 30
N_BOOT = 5000
BOOT_SEED = 20260813
CI_LOW = 2.5
CI_HIGH = 97.5

rng = np.random.default_rng(BOOT_SEED)


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def parameter_first_mean(data):
    """
    Compute the planned point estimator:
    mean over seeds for each parameter, then mean over parameters.
    """
    param_means = (
        data.groupby("param_index", observed=True)["Q_k"]
        .mean()
    )
    return float(param_means.mean())


def make_param_seed_matrix(data):
    """
    Build a parameter x seed matrix for seed-level bootstrap.
    """
    matrix = data.pivot_table(
        index="param_index",
        columns="seed",
        values="Q_k",
        aggfunc="mean"
    )

    return matrix.sort_index(axis=0).sort_index(axis=1)


def bootstrap_mean_q(data, n_boot, rng):
    """
    Bootstrap the main estimator by resampling seeds with replacement.

    For each bootstrap replicate:
      1. Resample seed columns.
      2. Mean across sampled seeds for each parameter.
      3. Mean across parameters.
    """
    matrix = make_param_seed_matrix(data)

    if matrix.shape[1] < 2:
        return np.array([], dtype=float)

    values = matrix.to_numpy(dtype=float)
    n_seeds = values.shape[1]

    boot_values = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        sampled_cols = rng.integers(0, n_seeds, size=n_seeds)
        sampled = values[:, sampled_cols]

        param_means = np.nanmean(sampled, axis=1)
        boot_values[b] = np.nanmean(param_means)

    return boot_values


def linear_fit_with_r2(x, y):
    """
    Fit y = slope * x + intercept and return slope, intercept, R^2.
    """
    slope, intercept = np.polyfit(x, y, 1)

    pred = slope * x + intercept

    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = np.nan

    return float(slope), float(intercept), float(r2)


def bootstrap_scaling_fit(group_data, n_boot, rng):
    """
    Bootstrap the log-linear slope across n.

    Seeds are resampled independently within each n condition.
    The estimator at each n remains parameter-first.
    """
    n_values = sorted(group_data["n_qubits"].unique())

    matrices = {}

    for n in n_values:
        sub = group_data[group_data["n_qubits"] == n]
        matrix = make_param_seed_matrix(sub)

        if matrix.shape[1] < 2:
            return np.array([], dtype=float)

        matrices[n] = matrix.to_numpy(dtype=float)

    boot_slopes = []

    for _ in range(n_boot):
        x_boot = []
        y_boot = []

        valid = True

        for n in n_values:
            values = matrices[n]
            n_seeds = values.shape[1]

            sampled_cols = rng.integers(0, n_seeds, size=n_seeds)
            sampled = values[:, sampled_cols]

            param_means = np.nanmean(sampled, axis=1)
            mean_q = np.nanmean(param_means)

            if not np.isfinite(mean_q) or mean_q <= 0:
                valid = False
                break

            x_boot.append(float(n))
            y_boot.append(np.log(mean_q))

        if valid and len(x_boot) >= 2:
            slope, _ = np.polyfit(
                np.asarray(x_boot),
                np.asarray(y_boot),
                1
            )
            boot_slopes.append(float(slope))

    return np.asarray(boot_slopes, dtype=float)


# ------------------------------------------------------------
# Load and validate data
# ------------------------------------------------------------

df = pd.read_csv(CSV_PATH)

required_q = [
    "n_qubits",
    "depth",
    "circuit_variant",
    "seed",
    "param_index",
    "abs_sum_k",
    "N_eff_k",
]

required_bridge = [
    "grad_k",
    "B_eff_k",
]

missing_q = [c for c in required_q if c not in df.columns]

if missing_q:
    raise ValueError(
        f"Missing columns required for Q analysis: {missing_q}"
    )

df["circuit_variant"] = (
    df["circuit_variant"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ------------------------------------------------------------
# Prepare Q_k analysis data
# ------------------------------------------------------------

q_df = df[required_q].copy()

q_df = q_df.dropna(
    subset=[
        "n_qubits",
        "depth",
        "circuit_variant",
        "seed",
        "param_index",
        "abs_sum_k",
        "N_eff_k",
    ]
)

q_df = q_df[q_df["N_eff_k"] > 0].copy()

q_df["Q_k"] = (
    q_df["abs_sum_k"] ** 2
    / q_df["N_eff_k"]
)

if not np.all(np.isfinite(q_df["Q_k"])):
    raise ValueError("Non-finite Q_k values detected.")


# ------------------------------------------------------------
# Check duplicate rows
# ------------------------------------------------------------

key_cols = [
    "n_qubits",
    "depth",
    "circuit_variant",
    "seed",
    "param_index",
]

duplicate_count = q_df.duplicated(
    subset=key_cols,
    keep=False
).sum()

print(f"Duplicate parameter-seed rows: {duplicate_count}")

if duplicate_count > 0:
    warnings.warn(
        "Duplicate parameter-seed rows detected. "
        "Pivot tables will average duplicate values."
    )


# ------------------------------------------------------------
# Check seed coverage
# ------------------------------------------------------------

seed_coverage = (
    q_df.groupby(
        ["n_qubits", "depth", "circuit_variant", "param_index"],
        observed=True
    )["seed"]
    .nunique()
    .reset_index(name="n_unique_seeds")
)

print("\nSeed coverage summary:")
print(
    seed_coverage["n_unique_seeds"]
    .value_counts()
    .sort_index()
    .to_string()
)

incomplete = seed_coverage[
    seed_coverage["n_unique_seeds"] != EXPECTED_SEEDS
]

if len(incomplete) > 0:
    warnings.warn(
        f"{len(incomplete)} parameter groups do not have "
        f"{EXPECTED_SEEDS} unique seeds."
    )

    print("\nIncomplete seed groups:")
    print(incomplete.to_string(index=False))


# ------------------------------------------------------------
# Exact bridge consistency check
# ------------------------------------------------------------

if all(c in df.columns for c in required_bridge):
    bridge_cols = required_q + required_bridge

    bridge_df = df[bridge_cols].copy()

    bridge_df = bridge_df.dropna(
        subset=[
            "abs_sum_k",
            "N_eff_k",
            "grad_k",
            "B_eff_k",
        ]
    )

    bridge_df = bridge_df[
        bridge_df["N_eff_k"] > 0
    ].copy()

    bridge_df["Q_k"] = (
        bridge_df["abs_sum_k"] ** 2
        / bridge_df["N_eff_k"]
    )

    bridge_error = (
        bridge_df["B_eff_k"] ** 2 * bridge_df["Q_k"]
        - bridge_df["grad_k"] ** 2
    )

    abs_error = bridge_error.abs()

    scale = np.maximum(
        bridge_df["grad_k"] ** 2,
        1e-15
    )

    rel_error = abs_error / scale

    print("\nBridge consistency:")
    print(
        f"  max absolute error    = {abs_error.max():.3e}"
    )
    print(
        f"  median absolute error = {abs_error.median():.3e}"
    )
    print(
        f"  median relative error = {rel_error.median():.3e}"
    )
    print(
        f"  99th pct rel. error   = "
        f"{np.percentile(rel_error, 99):.3e}"
    )

else:
    warnings.warn(
        "grad_k or B_eff_k is missing. "
        "Bridge consistency check skipped."
    )


# ------------------------------------------------------------
# Point estimates and bootstrap confidence intervals
# ------------------------------------------------------------

summary_rows = []

group_cols = [
    "n_qubits",
    "depth",
    "circuit_variant",
]

for keys, sub in q_df.groupby(group_cols, observed=True):
    n_qubits, depth, variant = keys

    point_mean = parameter_first_mean(sub)

    param_means = (
        sub.groupby("param_index", observed=True)["Q_k"]
        .mean()
    )

    n_params = int(param_means.shape[0])
    n_unique_seeds = int(sub["seed"].nunique())

    boot_values = bootstrap_mean_q(
        sub,
        n_boot=N_BOOT,
        rng=rng
    )

    if boot_values.size > 0:
        ci_low = float(
            np.percentile(boot_values, CI_LOW)
        )
        ci_high = float(
            np.percentile(boot_values, CI_HIGH)
        )
        boot_sd = float(np.std(boot_values, ddof=1))
    else:
        ci_low = np.nan
        ci_high = np.nan
        boot_sd = np.nan

    summary_rows.append({
        "n_qubits": n_qubits,
        "depth": depth,
        "circuit_variant": variant,
        "mean_Q": point_mean,
        "bootstrap_sd_Q": boot_sd,
        "ci_low_Q": ci_low,
        "ci_high_Q": ci_high,
        "n_params": n_params,
        "n_unique_seeds": n_unique_seeds,
    })

summary = pd.DataFrame(summary_rows)

summary = summary.sort_values(
    ["depth", "circuit_variant", "n_qubits"]
)

summary.to_csv(
    "q_scaling.csv",
    index=False
)

print("\nQ scaling summary:")
print(summary.to_string(index=False))


# ------------------------------------------------------------
# Log-linear scaling fits
# ------------------------------------------------------------

print(
    "\nLog-linear fits: "
    "log(mean_Q) = slope * n + intercept"
)

fit_rows = []

for (depth, variant), sub_summary in summary.groupby(
    ["depth", "circuit_variant"],
    observed=True
):
    sub_summary = sub_summary.sort_values("n_qubits")

    x = sub_summary["n_qubits"].to_numpy(dtype=float)
    mean_q = sub_summary["mean_Q"].to_numpy(dtype=float)

    valid = np.isfinite(mean_q) & (mean_q > 0)

    if valid.sum() < 2:
        continue

    x_valid = x[valid]
    log_q = np.log(mean_q[valid])

    slope, intercept, r2 = linear_fit_with_r2(
        x_valid,
        log_q
    )

    factor = float(np.exp(slope))

    raw_sub = q_df[
        (q_df["depth"] == depth)
        & (q_df["circuit_variant"] == variant)
        & (q_df["n_qubits"].isin(x_valid))
    ].copy()

    boot_slopes = bootstrap_scaling_fit(
        raw_sub,
        n_boot=N_BOOT,
        rng=rng
    )

    if boot_slopes.size > 0:
        slope_ci_low = float(
            np.percentile(boot_slopes, CI_LOW)
        )
        slope_ci_high = float(
            np.percentile(boot_slopes, CI_HIGH)
        )

        factor_ci_low = float(
            np.exp(slope_ci_low)
        )
        factor_ci_high = float(
            np.exp(slope_ci_high)
        )
    else:
        slope_ci_low = np.nan
        slope_ci_high = np.nan
        factor_ci_low = np.nan
        factor_ci_high = np.nan

    fit_rows.append({
        "depth": depth,
        "variant": variant,
        "n_points": int(valid.sum()),
        "slope": slope,
        "slope_ci_low": slope_ci_low,
        "slope_ci_high": slope_ci_high,
        "intercept": intercept,
        "r2": r2,
        "decay_factor_per_qubit": factor,
        "factor_ci_low": factor_ci_low,
        "factor_ci_high": factor_ci_high,
    })

    print(
        f"depth={depth}, variant={variant}: "
        f"slope={slope:+.4f} "
        f"[{slope_ci_low:+.4f}, {slope_ci_high:+.4f}], "
        f"factor={factor:.4f}, "
        f"R2={r2:.4f}"
    )

fit_df = pd.DataFrame(fit_rows)

fit_df.to_csv(
    "q_scaling_fits.csv",
    index=False
)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

depths = sorted(summary["depth"].unique())

fig, axes = plt.subplots(
    1,
    len(depths),
    figsize=(5.0 * len(depths), 4.2),
    sharey=True
)

if len(depths) == 1:
    axes = [axes]

style = {
    "hea": {
        "color": "tab:blue",
        "marker": "s",
        "label": "HEA",
    },
    "hva": {
        "color": "tab:orange",
        "marker": "o",
        "label": "HVA",
    },
}

for ax, depth in zip(axes, depths):
    depth_data = summary[
        summary["depth"] == depth
    ]

    variants = sorted(
        depth_data["circuit_variant"].unique()
    )

    for variant in variants:
        sub = depth_data[
            depth_data["circuit_variant"] == variant
        ].sort_values("n_qubits")

        if len(sub) == 0:
            continue

        if variant in style:
            plot_style = style[variant]
        else:
            plot_style = {
                "color": None,
                "marker": "o",
                "label": variant.upper(),
            }

        y = sub["mean_Q"].to_numpy(dtype=float)
        y_low = sub["ci_low_Q"].to_numpy(dtype=float)
        y_high = sub["ci_high_Q"].to_numpy(dtype=float)

        lower_err = y - y_low
        upper_err = y_high - y

        yerr = np.vstack([
            lower_err,
            upper_err,
        ])

        ax.errorbar(
            sub["n_qubits"],
            y,
            yerr=yerr,
            color=plot_style["color"],
            marker=plot_style["marker"],
            markersize=7,
            linewidth=2.0,
            capsize=4,
            label=plot_style["label"],
        )

    ax.set_yscale("log")
    ax.set_xlabel(
        "Number of qubits",
        fontsize=12
    )

    ax.set_xticks(
        sorted(summary["n_qubits"].unique())
    )

    ax.tick_params(
        axis="both",
        labelsize=11
    )

    ax.grid(
        True,
        which="both",
        alpha=0.3
    )

    ax.set_title(
        f"depth = {depth}",
        fontsize=12
    )

axes[0].set_ylabel(
    "Mean activity scale Q",
    fontsize=12
)

axes[-1].legend(
    frameon=True,
    fontsize=10
)

plt.tight_layout()

fig.savefig(
    "q_scaling.pdf",
    bbox_inches="tight"
)

plt.show()
