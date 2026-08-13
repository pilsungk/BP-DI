# ============================================================
# Step 2: E[B_eff^2] scaling analysis
#
# Quantity:
#   B2_k = B_eff_k^2
#
# Theoretical references:
#   Random-sign conditional second moment:
#       E_s[B2_k | magnitudes] = 1
#
#   Fixed-magnitude organized upper bound:
#       B2_k <= N_eff_k
#
# Main estimator:
#   1. Mean over seeds for each parameter
#   2. Mean over parameters
#
# Uncertainty:
#   Bootstrap over seeds as the independent resampling unit.
#
# Important interpretation:
#   Proximity to B2 = 1 is descriptive consistency evidence
#   for the random-sign second-moment baseline, not a formal
#   random-sign hypothesis test.
#
# Outputs:
#   b2_scaling.csv
#   b2_scaling_fits.csv
#   b2_scaling.pdf
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

def parameter_first_mean(data, value_col):
    """
    Mean over seeds for each parameter, then mean over parameters.
    """
    param_means = (
        data.groupby("param_index", observed=True)[value_col]
        .mean()
    )

    return float(param_means.mean())


def make_param_seed_matrix(data, value_col):
    """
    Build parameter x seed matrix.
    """
    matrix = data.pivot_table(
        index="param_index",
        columns="seed",
        values=value_col,
        aggfunc="mean"
    )

    return matrix.sort_index(axis=0).sort_index(axis=1)


def bootstrap_mean(data, value_col, n_boot, rng):
    """
    Bootstrap the parameter-first estimator by resampling seeds.
    """
    matrix = make_param_seed_matrix(
        data,
        value_col
    )

    if matrix.shape[1] < 2:
        return np.array([], dtype=float)

    values = matrix.to_numpy(dtype=float)
    n_seeds = values.shape[1]

    boot_values = np.empty(
        n_boot,
        dtype=float
    )

    for b in range(n_boot):
        sampled_cols = rng.integers(
            0,
            n_seeds,
            size=n_seeds
        )

        sampled = values[:, sampled_cols]

        param_means = np.nanmean(
            sampled,
            axis=1
        )

        boot_values[b] = np.nanmean(
            param_means
        )

    return boot_values


def linear_fit_with_r2(x, y):
    """
    Fit y = slope * x + intercept.
    """
    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    pred = slope * x + intercept

    ss_res = np.sum(
        (y - pred) ** 2
    )

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    if ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = np.nan

    return (
        float(slope),
        float(intercept),
        float(r2)
    )


def bootstrap_scaling_fit(
    group_data,
    value_col,
    n_boot,
    rng
):
    """
    Bootstrap log-linear slope across n.

    Seeds are resampled within each n condition.
    """
    n_values = sorted(
        group_data["n_qubits"].unique()
    )

    matrices = {}

    for n in n_values:
        sub = group_data[
            group_data["n_qubits"] == n
        ]

        matrix = make_param_seed_matrix(
            sub,
            value_col
        )

        if matrix.shape[1] < 2:
            return np.array([], dtype=float)

        matrices[n] = matrix.to_numpy(
            dtype=float
        )

    boot_slopes = []

    for _ in range(n_boot):
        x_boot = []
        y_boot = []

        valid = True

        for n in n_values:
            values = matrices[n]
            n_seeds = values.shape[1]

            sampled_cols = rng.integers(
                0,
                n_seeds,
                size=n_seeds
            )

            sampled = values[:, sampled_cols]

            param_means = np.nanmean(
                sampled,
                axis=1
            )

            mean_value = np.nanmean(
                param_means
            )

            if (
                not np.isfinite(mean_value)
                or mean_value <= 0
            ):
                valid = False
                break

            x_boot.append(
                float(n)
            )

            y_boot.append(
                np.log(mean_value)
            )

        if valid and len(x_boot) >= 2:
            slope, _ = np.polyfit(
                np.asarray(x_boot),
                np.asarray(y_boot),
                1
            )

            boot_slopes.append(
                float(slope)
            )

    return np.asarray(
        boot_slopes,
        dtype=float
    )


# ------------------------------------------------------------
# Load and validate data
# ------------------------------------------------------------

df = pd.read_csv(CSV_PATH)

required = [
    "n_qubits",
    "depth",
    "circuit_variant",
    "seed",
    "param_index",
    "B_eff_k",
    "N_eff_k",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

df["circuit_variant"] = (
    df["circuit_variant"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ------------------------------------------------------------
# Prepare Step 2 data
# ------------------------------------------------------------

n_before = len(df)

b_df = df[required].copy()

b_df = b_df.dropna(
    subset=required
)

b_df = b_df[
    b_df["N_eff_k"] > 0
].copy()

n_after = len(b_df)

print(
    f"Rows before filtering: {n_before}, "
    f"after: {n_after}, "
    f"dropped: {n_before - n_after}"
)

b_df["B2_k"] = (
    b_df["B_eff_k"] ** 2
)

# Row-level fraction of the fixed-magnitude upper bound.
# Since B2_k <= N_eff_k, this should ideally lie in [0, 1]
# up to numerical precision.
b_df["organization_fraction_k"] = (
    b_df["B2_k"]
    / b_df["N_eff_k"]
)

if not np.all(
    np.isfinite(b_df["B2_k"])
):
    raise ValueError(
        "Non-finite B2_k values detected."
    )

if not np.all(
    np.isfinite(
        b_df["organization_fraction_k"]
    )
):
    raise ValueError(
        "Non-finite organization fractions detected."
    )


# ------------------------------------------------------------
# Duplicate check
# ------------------------------------------------------------

key_cols = [
    "n_qubits",
    "depth",
    "circuit_variant",
    "seed",
    "param_index",
]

duplicate_count = b_df.duplicated(
    subset=key_cols,
    keep=False
).sum()

print(
    f"Duplicate parameter-seed rows: "
    f"{duplicate_count}"
)

if duplicate_count > 0:
    warnings.warn(
        "Duplicate parameter-seed rows detected. "
        "Pivot tables will average duplicates."
    )


# ------------------------------------------------------------
# Seed coverage check
# ------------------------------------------------------------

seed_coverage = (
    b_df.groupby(
        [
            "n_qubits",
            "depth",
            "circuit_variant",
            "param_index",
        ],
        observed=True
    )["seed"]
    .nunique()
    .reset_index(
        name="n_unique_seeds"
    )
)

print("\nSeed coverage summary:")

print(
    seed_coverage["n_unique_seeds"]
    .value_counts()
    .sort_index()
    .to_string()
)

incomplete = seed_coverage[
    seed_coverage["n_unique_seeds"]
    != EXPECTED_SEEDS
]

if len(incomplete) > 0:
    warnings.warn(
        f"{len(incomplete)} parameter groups "
        f"do not have {EXPECTED_SEEDS} unique seeds."
    )

    print("\nIncomplete seed groups:")
    print(
        incomplete.to_string(
            index=False
        )
    )


# ------------------------------------------------------------
# Numerical upper-bound check
# ------------------------------------------------------------

bound_violation = (
    b_df["B2_k"]
    - b_df["N_eff_k"]
)

max_bound_violation = float(
    bound_violation.max()
)

n_bound_violations = int(
    np.sum(
        bound_violation > 1e-10
    )
)

print("\nFixed-magnitude bound check:")
print(
    f"  max(B2 - N_eff) = "
    f"{max_bound_violation:.3e}"
)
print(
    f"  violations > 1e-10 = "
    f"{n_bound_violations}"
)


# ------------------------------------------------------------
# Summary statistics and bootstrap CIs
# ------------------------------------------------------------

summary_rows = []

group_cols = [
    "n_qubits",
    "depth",
    "circuit_variant",
]

for keys, sub in b_df.groupby(
    group_cols,
    observed=True
):
    n_qubits, depth, variant = keys

    # Main B2 estimator
    point_b2 = parameter_first_mean(
        sub,
        "B2_k"
    )

    boot_b2 = bootstrap_mean(
        sub,
        "B2_k",
        N_BOOT,
        rng
    )

    if boot_b2.size > 0:
        b2_ci_low = float(
            np.percentile(
                boot_b2,
                CI_LOW
            )
        )

        b2_ci_high = float(
            np.percentile(
                boot_b2,
                CI_HIGH
            )
        )

        b2_boot_sd = float(
            np.std(
                boot_b2,
                ddof=1
            )
        )
    else:
        b2_ci_low = np.nan
        b2_ci_high = np.nan
        b2_boot_sd = np.nan

    # Descriptive consistency with random-sign baseline = 1.
    # This is not a formal random-sign hypothesis test.
    baseline_difference = (
        point_b2 - 1.0
    )

    ci_contains_1 = bool(
        np.isfinite(b2_ci_low)
        and np.isfinite(b2_ci_high)
        and b2_ci_low <= 1.0 <= b2_ci_high
    )

    # N_eff context
    point_neff = parameter_first_mean(
        sub,
        "N_eff_k"
    )

    # Ratio of means, retained only as descriptive context.
    ratio_of_means = (
        point_b2 / point_neff
        if point_neff > 0
        else np.nan
    )

    # Preferred organization statistic:
    # row-level B2 / N_eff followed by the same estimator.
    point_org_fraction = parameter_first_mean(
        sub,
        "organization_fraction_k"
    )

    boot_org = bootstrap_mean(
        sub,
        "organization_fraction_k",
        N_BOOT,
        rng
    )

    if boot_org.size > 0:
        org_ci_low = float(
            np.percentile(
                boot_org,
                CI_LOW
            )
        )

        org_ci_high = float(
            np.percentile(
                boot_org,
                CI_HIGH
            )
        )
    else:
        org_ci_low = np.nan
        org_ci_high = np.nan

    summary_rows.append({
        "n_qubits": n_qubits,
        "depth": depth,
        "circuit_variant": variant,

        "mean_B2": point_b2,
        "bootstrap_sd_B2": b2_boot_sd,
        "ci_low_B2": b2_ci_low,
        "ci_high_B2": b2_ci_high,

        "B2_minus_random_baseline": baseline_difference,
        "ci_contains_random_baseline_1": ci_contains_1,

        "mean_N_eff": point_neff,
        "ratio_of_means_B2_over_Neff": ratio_of_means,

        "mean_organization_fraction": point_org_fraction,
        "ci_low_organization_fraction": org_ci_low,
        "ci_high_organization_fraction": org_ci_high,

        "n_params": int(
            sub["param_index"].nunique()
        ),
        "n_unique_seeds": int(
            sub["seed"].nunique()
        ),
    })


summary = pd.DataFrame(
    summary_rows
)

summary = summary.sort_values(
    [
        "depth",
        "circuit_variant",
        "n_qubits",
    ]
)

summary.to_csv(
    "b2_scaling.csv",
    index=False
)

print("\nB2 scaling summary:")
print(
    summary.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Log-linear scaling fits
# ------------------------------------------------------------

print(
    "\nLog-linear fits: "
    "log(mean_B2) = slope * n + intercept"
)

fit_rows = []

for (depth, variant), sub_summary in summary.groupby(
    [
        "depth",
        "circuit_variant",
    ],
    observed=True
):
    sub_summary = sub_summary.sort_values(
        "n_qubits"
    )

    x = sub_summary[
        "n_qubits"
    ].to_numpy(
        dtype=float
    )

    mean_b2 = sub_summary[
        "mean_B2"
    ].to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(mean_b2)
        & (mean_b2 > 0)
    )

    if valid.sum() < 2:
        continue

    x_valid = x[valid]
    log_b2 = np.log(
        mean_b2[valid]
    )

    slope, intercept, r2 = (
        linear_fit_with_r2(
            x_valid,
            log_b2
        )
    )

    factor = float(
        np.exp(slope)
    )

    raw_sub = b_df[
        (b_df["depth"] == depth)
        & (
            b_df["circuit_variant"]
            == variant
        )
        & (
            b_df["n_qubits"]
            .isin(x_valid)
        )
    ].copy()

    boot_slopes = bootstrap_scaling_fit(
        raw_sub,
        "B2_k",
        N_BOOT,
        rng
    )

    if boot_slopes.size > 0:
        slope_ci_low = float(
            np.percentile(
                boot_slopes,
                CI_LOW
            )
        )

        slope_ci_high = float(
            np.percentile(
                boot_slopes,
                CI_HIGH
            )
        )

        factor_ci_low = float(
            np.exp(
                slope_ci_low
            )
        )

        factor_ci_high = float(
            np.exp(
                slope_ci_high
            )
        )
    else:
        slope_ci_low = np.nan
        slope_ci_high = np.nan
        factor_ci_low = np.nan
        factor_ci_high = np.nan

    fit_rows.append({
        "depth": depth,
        "variant": variant,
        "n_points": int(
            valid.sum()
        ),
        "slope": slope,
        "slope_ci_low": slope_ci_low,
        "slope_ci_high": slope_ci_high,
        "intercept": intercept,
        "r2": r2,
        "factor_per_qubit": factor,
        "factor_ci_low": factor_ci_low,
        "factor_ci_high": factor_ci_high,
    })

    print(
        f"depth={depth}, "
        f"variant={variant}: "
        f"slope={slope:+.4f} "
        f"[{slope_ci_low:+.4f}, "
        f"{slope_ci_high:+.4f}], "
        f"factor={factor:.4f}, "
        f"R2={r2:.4f}"
    )


fit_df = pd.DataFrame(
    fit_rows
)

fit_df.to_csv(
    "b2_scaling_fits.csv",
    index=False
)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

depths = sorted(
    summary["depth"].unique()
)

fig, axes = plt.subplots(
    1,
    len(depths),
    figsize=(
        5.0 * len(depths),
        4.2
    ),
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

for ax, depth in zip(
    axes,
    depths
):
    depth_data = summary[
        summary["depth"] == depth
    ]

    variants = sorted(
        depth_data[
            "circuit_variant"
        ].unique()
    )

    for variant in variants:
        sub = depth_data[
            depth_data[
                "circuit_variant"
            ] == variant
        ].sort_values(
            "n_qubits"
        )

        if len(sub) == 0:
            continue

        plot_style = style.get(
            variant,
            {
                "color": None,
                "marker": "o",
                "label": variant.upper(),
            }
        )

        y = sub[
            "mean_B2"
        ].to_numpy(
            dtype=float
        )

        y_low = sub[
            "ci_low_B2"
        ].to_numpy(
            dtype=float
        )

        y_high = sub[
            "ci_high_B2"
        ].to_numpy(
            dtype=float
        )

        lower_err = np.maximum(
            y - y_low,
            0.0
        )

        upper_err = np.maximum(
            y_high - y,
            0.0
        )

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

    # Random-sign exact conditional second-moment reference.
    ax.axhline(
        1.0,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Random-sign reference"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "Number of qubits",
        fontsize=12
    )

    ax.set_xticks(
        sorted(
            summary[
                "n_qubits"
            ].unique()
        )
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
    "Mean squared interference quality",
    fontsize=12
)

# Avoid duplicate legend labels.
handles, labels = axes[-1].get_legend_handles_labels()

unique = {}

for handle, label in zip(
    handles,
    labels
):
    if label not in unique:
        unique[label] = handle

axes[-1].legend(
    unique.values(),
    unique.keys(),
    frameon=True,
    fontsize=10
)

plt.tight_layout()

fig.savefig(
    "b2_scaling.pdf",
    bbox_inches="tight"
)

plt.show()
