# ============================================================
# Step 4: Within/between-seed decomposition of
# B2-Q coupling
#
# Motivation:
#   F - 1 = Cov(B2, Q) / (E[B2] E[Q]),
#   and by the law of total covariance (seed as conditioning
#   variable, plug-in ddof=0, balanced complete data):
#     Cov_total = E_s[Cov_k(B2,Q|s)] + Cov_s(mean_k B2, mean_k Q)
#                 (within-seed)        (between-seed)
#   The within and between components are therefore reported
#   in normalized form,
#     norm_within  = mean_s Cov_within / (E[B2] E[Q])
#     norm_between = Cov_between       / (E[B2] E[Q])
#   which sum exactly to F - 1.
#
# Statistical design (consistent with Steps 1-3):
#   - Seed is the independent resampling unit.
#   - Within-seed correlations: computed per seed across
#     parameters, summarized over 30 seeds.
#   - Between-seed correlation: across the 30 seed-level
#     parameter means.
#   - Joint bootstrap: one seed resample per replicate,
#     all quantities recomputed on the same replicate.
#   - Pooled correlations retained as descriptive reference.
#
# Internal consistency checks:
#   (a) decomposition residual
#       Cov_total - (mean Cov_within + Cov_between) ~ 0
#   (b) norm_within + norm_between = F - 1 (algebraic)
#
# Outputs:
#   lfim_b2_q_coupling_decomposition.csv
#   lfim_b2_q_coupling.pdf
# ============================================================
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
CSV_PATH = "../results/lfim/per_run.csv"
EXPECTED_SEEDS = 30
N_BOOT = 1000
BOOT_SEED = 20260813
CI_LOW = 2.5
CI_HIGH = 97.5
MIN_PARAMS_FOR_CORR = 3
REL_TOL = 1e-12

rng = np.random.default_rng(BOOT_SEED)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def make_param_seed_frame(data, value_col):
    frame = data.pivot_table(
        index="param_index",
        columns="seed",
        values=value_col,
        aggfunc="mean"
    )
    return frame.sort_index(axis=0).sort_index(axis=1)


def pearson_corr(x, y):
    """
    Pearson correlation with relative-tolerance guard
    against (near-)constant inputs.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < MIN_PARAMS_FOR_CORR:
        return np.nan
    sx = np.std(x)
    sy = np.std(y)
    tol_x = REL_TOL * max(1.0, float(np.max(np.abs(x))))
    tol_y = REL_TOL * max(1.0, float(np.max(np.abs(y))))
    if sx < tol_x or sy < tol_y:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def rank_transform(x):
    return pd.Series(x).rank(method="average").to_numpy(float)


def spearman_corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < MIN_PARAMS_FOR_CORR:
        return np.nan
    return pearson_corr(rank_transform(x), rank_transform(y))


def plugin_cov(x, y):
    """
    Plug-in (ddof=0) covariance. Required for exact
    additivity of the total-covariance decomposition.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return np.nan
    return float(np.mean(x * y) - np.mean(x) * np.mean(y))


def group_coupling_stats(mb2, mq, compute_spearman=True,
                         precomputed_pears_w=None):
    n_seeds = mb2.shape[1]

    # Within-seed Pearson: use precomputed values when given
    if precomputed_pears_w is not None:
        pears_w = precomputed_pears_w
    else:
        pears_w = np.array(
            [pearson_corr(mb2[:, s], mq[:, s])
             for s in range(n_seeds)], dtype=float)

    if compute_spearman:
        spear_w = np.array(
            [spearman_corr(mb2[:, s], mq[:, s])
             for s in range(n_seeds)], dtype=float)
        mean_spear_w = float(np.nanmean(spear_w))
    else:
        mean_spear_w = np.nan

    cov_w = np.array(
        [plugin_cov(mb2[:, s], mq[:, s])
         for s in range(n_seeds)], dtype=float)

    bbar = np.nanmean(mb2, axis=0)
    qbar = np.nanmean(mq, axis=0)
    pears_b = pearson_corr(bbar, qbar)
    spear_b = (spearman_corr(bbar, qbar)
               if compute_spearman else np.nan)
    cov_b = plugin_cov(bbar, qbar)

    all_b2 = mb2.ravel()
    all_q = mq.ravel()
    cov_total = plugin_cov(all_b2, all_q)
    e_b2 = float(np.nanmean(all_b2))
    e_q = float(np.nanmean(all_q))

    mean_cov_w = float(np.nanmean(cov_w))
    resid = cov_total - (mean_cov_w + cov_b)
    denom = e_b2 * e_q
    f_minus_1 = cov_total / denom if denom > 0 else np.nan
    norm_w = mean_cov_w / denom if denom > 0 else np.nan
    norm_b = cov_b / denom if denom > 0 else np.nan

    return {
        "mean_pears_w": float(np.nanmean(pears_w)),
        "pears_w_per_seed": pears_w,
        "mean_spear_w": mean_spear_w,
        "pears_b": pears_b,
        "spear_b": spear_b,
        "mean_cov_w": mean_cov_w,
        "cov_b": cov_b,
        "cov_total": cov_total,
        "decomp_residual": resid,
        "F_minus_1": f_minus_1,
        "norm_within": norm_w,
        "norm_between": norm_b,
    }


def joint_bootstrap_coupling(mb2, mq, pears_w_full,
                             n_boot, rng):
    """
    pears_w_full: per-seed within Pearson from the original
    data; replicates average the selected entries (exactly
    equivalent to recomputation, since each within-seed
    correlation depends only on that seed's column).
    Spearman is excluded from the bootstrap (point estimates
    only, reported in the CSV).
    """
    n_seeds = mb2.shape[1]
    keys = ["mean_pears_w", "pears_b",
            "norm_within", "norm_between", "F_minus_1"]
    if n_seeds < 2:
        empty = np.array([], dtype=float)
        return {k: empty for k in keys}
    out = {k: np.empty(n_boot, dtype=float) for k in keys}
    for b in range(n_boot):
        cols = rng.integers(0, n_seeds, size=n_seeds)
        st = group_coupling_stats(
            mb2[:, cols], mq[:, cols],
            compute_spearman=False,
            precomputed_pears_w=pears_w_full[cols],
        )
        for k in keys:
            out[k][b] = st[k]
    return out


def pctl_ci(arr):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(finite, CI_LOW)),
            float(np.percentile(finite, CI_HIGH)))

# ------------------------------------------------------------
# Load and prepare data
# ------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
required = [
    "n_qubits", "depth", "circuit_variant", "seed",
    "param_index", "B_eff_k", "abs_sum_k", "N_eff_k",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df["circuit_variant"] = (
    df["circuit_variant"].astype(str).str.strip().str.lower()
)

n_before = len(df)
w_df = df[required].dropna(subset=required).copy()
w_df = w_df[w_df["N_eff_k"] > 0].copy()
n_after = len(w_df)
print(f"Rows before filtering: {n_before}, after: {n_after}, "
      f"dropped: {n_before - n_after}")

w_df["Q_k"] = w_df["abs_sum_k"] ** 2 / w_df["N_eff_k"]
w_df["B2_k"] = w_df["B_eff_k"] ** 2

# ------------------------------------------------------------
# Per-group decomposition with joint bootstrap
# ------------------------------------------------------------
summary_rows = []
group_cols = ["n_qubits", "depth", "circuit_variant"]

for keys, sub in w_df.groupby(group_cols, observed=True):
    n_qubits, depth, variant = keys

    fb2 = make_param_seed_frame(sub, "B2_k")
    fq = make_param_seed_frame(sub, "Q_k")
    common_rows = fb2.index.intersection(fq.index)
    common_cols = fb2.columns.intersection(fq.columns)
    mb2 = fb2.loc[common_rows, common_cols].to_numpy(float)
    mq = fq.loc[common_rows, common_cols].to_numpy(float)

    st = group_coupling_stats(mb2, mq)   # full, with Spearman
    boot = joint_bootstrap_coupling(
        mb2, mq, st["pears_w_per_seed"], N_BOOT, rng
    )
    
    pw_lo, pw_hi = pctl_ci(boot["mean_pears_w"])
    pb_lo, pb_hi = pctl_ci(boot["pears_b"])
    nw_lo, nw_hi = pctl_ci(boot["norm_within"])
    nb_lo, nb_hi = pctl_ci(boot["norm_between"])

    # Descriptive pooled reference
    pooled_p = pearson_corr(
        sub["B2_k"].to_numpy(float),
        sub["Q_k"].to_numpy(float)
    )
    pooled_s = spearman_corr(
        sub["B2_k"].to_numpy(float),
        sub["Q_k"].to_numpy(float)
    )

    summary_rows.append({
        "n_qubits": n_qubits,
        "depth": depth,
        "circuit_variant": variant,
        # Within-seed
        "mean_within_pearson": st["mean_pears_w"],
        "ci_low_within_pearson": pw_lo,
        "ci_high_within_pearson": pw_hi,
        "mean_within_spearman": st["mean_spear_w"],

        # Between-seed
        "between_pearson": st["pears_b"],
        "ci_low_between_pearson": pb_lo,
        "ci_high_between_pearson": pb_hi,
        "between_spearman": st["spear_b"],

        # Covariance decomposition (normalized so the two
        # components sum to F - 1)
        "F_minus_1": st["F_minus_1"],
        "norm_within": st["norm_within"],
        "ci_low_norm_within": nw_lo,
        "ci_high_norm_within": nw_hi,
        "norm_between": st["norm_between"],
        "ci_low_norm_between": nb_lo,
        "ci_high_norm_between": nb_hi,
        "decomp_residual": st["decomp_residual"],
        # Reference
        "pooled_pearson_ref": pooled_p,
        "pooled_spearman_ref": pooled_s,
        "n_params": mb2.shape[0],
        "n_seeds": mb2.shape[1],
    })

summary = pd.DataFrame(summary_rows)
summary = summary.sort_values(
    ["depth", "circuit_variant", "n_qubits"]
)
summary.to_csv("lfim_b2_q_coupling_decomposition.csv",
               index=False)

print("\nB2-Q coupling decomposition summary:")
show_cols = [
    "n_qubits", "depth", "circuit_variant",
    "mean_within_pearson", "ci_low_within_pearson",
    "ci_high_within_pearson",
    "between_pearson", "ci_low_between_pearson",
    "ci_high_between_pearson",
    "F_minus_1", "norm_within", "norm_between",
    "decomp_residual",
]
print(summary[show_cols].to_string(index=False))

# Consistency check: norm_within + norm_between vs F - 1
check = (summary["norm_within"] + summary["norm_between"]
         - summary["F_minus_1"]).abs()
print(f"\nMax |norm_within + norm_between - (F-1)|: "
      f"{check.max():.3e}")
print(f"Max |decomposition residual|: "
      f"{summary['decomp_residual'].abs().max():.3e}")

# ------------------------------------------------------------
# Figure: 2 rows (within-seed, between-seed Pearson with CI)
# x depths columns
# ------------------------------------------------------------
depths = sorted(summary["depth"].unique())
variants = ["hea", "hva"]
style = {
    "hea": {"color": "tab:blue", "marker": "s",
            "label": "HEA"},
    "hva": {"color": "tab:orange", "marker": "o",
            "label": "HVA"},
}
row_specs = [
    ("mean_within_pearson", "ci_low_within_pearson",
     "ci_high_within_pearson",
     "Mean within-seed corr(B2, Q)"),
    ("between_pearson", "ci_low_between_pearson",
     "ci_high_between_pearson",
     "Between-seed corr(B2, Q)"),
]

fig, axes = plt.subplots(
    2, len(depths),
    figsize=(5.0 * len(depths), 7.6),
    sharex="col", sharey="row"
)
axes = np.atleast_2d(axes)

for r, (col, lo_col, hi_col, ylabel) in enumerate(row_specs):
    for c, depth in enumerate(depths):
        ax = axes[r, c]
        sub_d = summary[summary["depth"] == depth]
        for variant in variants:
            sub = sub_d[
                sub_d["circuit_variant"] == variant
            ].sort_values("n_qubits")
            if len(sub) == 0:
                continue
            st = style[variant]
            y = sub[col].to_numpy(float)
            lo = sub[lo_col].to_numpy(float)
            hi = sub[hi_col].to_numpy(float)
            yerr = np.vstack([np.maximum(y - lo, 0),
                              np.maximum(hi - y, 0)])
            ax.errorbar(
                sub["n_qubits"], y, yerr=yerr,
                color=st["color"], marker=st["marker"],
                markersize=7, linewidth=2.0, capsize=4,
                label=st["label"]
            )
        ax.axhline(0.0, color="gray", linestyle=":",
                   linewidth=1.2)
        ax.set_xticks(sorted(summary["n_qubits"].unique()))
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.3)
        if r == 0:
            ax.set_title(f"depth = {depth}", fontsize=12)
        if r == len(row_specs) - 1:
            ax.set_xlabel("Number of qubits", fontsize=12)
        if c == 0:
            ax.set_ylabel(ylabel, fontsize=11)

h, l = axes[0, -1].get_legend_handles_labels()
uniq = dict(zip(l, h))
axes[0, -1].legend(uniq.values(), uniq.keys(),
                   frameon=True, fontsize=10)
plt.tight_layout()
fig.savefig("lfim_b2_q_coupling.pdf", bbox_inches="tight")
plt.show()
