# ============================================================
# Step 5 (sweep version): Sign-correlation structure from raw
# termwise data, with ZERO_TOL sensitivity sweep and
# N_PERM = 1000 permutation null.
#
# Estimand (agreed in the plan):
#   For each seed s, the within-circuit, parameter-averaged
#   sign-alignment matrix
#     C^(s)_{ab} = (1/K) sum_k sgn(a_{a,k}) sgn(a_{b,k})
#   The 30 seed-level matrices are the independent replicates.
#
# Zero handling:
#   Entries with |a| < zero_tol get sign 0. The same rule is
#   applied to permutation surrogates. The analysis is run at
#   every threshold in ZERO_TOL_LIST (sensitivity sweep).
#   Stage A additionally checks the magnitude distribution at
#   theoretically-zero positions to justify the final choice.
#
# Aggregate statistics (per seed -> mean over seeds):
#   1. mean absolute off-diagonal alignment
#   2. blockwise signed AND absolute means:
#      ZZ-ZZ, X-X, ZZ-X (absolute means guard against
#      within-sector cancellation of mixed +/- organization)
#   3. leading eigenvalue of C^(s), reported also as excess
#      and ratio relative to the permutation-null median
#      (raw values are not comparable across n because the
#      term count and zero pattern change with n)
#
# Uncertainty and null:
#   - Observed CI: bootstrap over the 30 per-seed statistics.
#   - Permutation null: entrywise independent Rademacher sign
#     flips of the sign matrices, preserving magnitudes and
#     the zero pattern. This realizes a conditional iid
#     random-sign null: it removes all term-term alignment
#     AND any per-term marginal sign imbalance. Comparison
#     against this null is therefore a CONSISTENCY TEST with
#     the random-sign model, not a pure test of pairwise
#     independence: an observed excess may originate from
#     pairwise correlations or from marginal sign bias.
#     The same seed-averaged statistics are recomputed on
#     N_PERM surrogate datasets to form the null distribution.
#
# Term ordering (verified against the experiment script):
#   with M = 2n - 1 terms, indices 0..n-2 are the ZZ edge
#   terms and indices n-1..2n-2 are the X field terms.
#
# HVA parameter ordering (verified against the experiment
# script): [b1, g1, b2, g2, ..., bd, gd]. The final gate
# block is the last layer RZZ, which commutes with every ZZ
# term, so d<ZZ_i>/d(gamma_last) = 0 analytically.
# Theoretical zero positions: ZZ rows 0..n-2, last parameter
# column.
#
# Outputs:
#   hva_theoretical_zero_magnitudes.csv
#   sign_correlation_summary_tol{tag}.csv   (per threshold)
#   sign_zero_fractions_tol{tag}.csv        (per threshold)
#   threshold_stability_comparison.csv
#   C_mean_abs_offdiag.pdf
#   C_blockwise.pdf
#   C_spectral.pdf
# ============================================================
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
RAW_DIR = "../data/raw_runs_tfim"

N_LIST = [4, 6, 8, 10]
D_LIST = [4, 6, 8]
VARIANTS = ["hea", "hva"]
SEEDS = list(range(30))
NPZ_KEY = "grad_terms"

ZERO_TOL_LIST = [1e-12, 1e-10, 1e-9, 1e-8]
N_PERM = 1000
N_BOOT = 2000
BOOT_SEED = 20260813
CI_LOW = 2.5
CI_HIGH = 97.5

# Threshold used for the final figures. Revisit after the
# Stage A gap check and the stability comparison.
FINAL_TOL_TAG = "1e-12"

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def npz_path(n, d, variant, seed):
    return os.path.join(
        RAW_DIR,
        f"n{n}_d{d}_{variant}_seed{seed:02d}_grad_terms.npz"
    )


def load_raw_matrix(n, d, variant, seed):
    """
    Load grad_terms (n_params x n_terms) and transpose to
    (n_terms x n_params). Returns None if the file is missing.
    """
    path = npz_path(n, d, variant, seed)
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        a = np.asarray(z[NPZ_KEY], dtype=float)
    return a.T


def signs_from_raw(a, zero_tol):
    """
    Convert a raw matrix to signs with the given threshold.
    """
    s = np.sign(a)
    s[np.abs(a) < zero_tol] = 0.0
    return s


def sector_indices(n, m_terms):
    """
    Term ordering: first n-1 terms are ZZ edges, last n terms
    are X fields (verified against the experiment script).
    """
    if m_terms != 2 * n - 1:
        warnings.warn(
            f"n={n}: expected {2*n-1} terms, got {m_terms}. "
            "Sector assignment may be wrong."
        )
    zz = np.arange(0, n - 1)
    x = np.arange(n - 1, m_terms)
    return zz, x


def alignment_matrix(s):
    """
    C = (1/K) S S^T from the sign matrix S (M x K).
    """
    k = s.shape[1]
    return (s @ s.T) / float(k)


def block_masks(m_terms, zz, x):
    """
    Boolean masks over the strict upper triangle for the
    three sector blocks.
    """
    iu = np.triu_indices(m_terms, k=1)
    in_zz = np.isin(iu[0], zz) & np.isin(iu[1], zz)
    in_x = np.isin(iu[0], x) & np.isin(iu[1], x)
    cross = (~in_zz) & (~in_x)
    return iu, in_zz, in_x, cross


STAT_KEYS = ["mean_abs_offdiag",
             "block_zz", "block_x", "block_cross",
             "block_zz_abs", "block_x_abs", "block_cross_abs",
             "lam_max"]

# Signed block statistics are physically meaningful in BOTH
# directions (negative values indicate anti-alignment, the
# structured anti-correlation route), so they get two-sided
# null flags. All other statistics are nonnegative-excess
# quantities and keep the upper-tail flag only.
TWO_SIDED_KEYS = ["block_zz", "block_x", "block_cross"]

def seed_statistics(c, iu, in_zz, in_x, cross):
    """
    Aggregate statistics from one seed-level matrix C.
    """
    off = c[iu]

    def m(v):
        return float(np.mean(v)) if v.size > 0 else np.nan

    return {
        "mean_abs_offdiag": float(np.mean(np.abs(off))),
        "block_zz": m(off[in_zz]),
        "block_x": m(off[in_x]),
        "block_cross": m(off[cross]),
        "block_zz_abs": m(np.abs(off[in_zz])),
        "block_x_abs": m(np.abs(off[in_x])),
        "block_cross_abs": m(np.abs(off[cross])),
        "lam_max": float(np.linalg.eigvalsh(c)[-1]),
    }


def dataset_statistics(sign_mats, iu, in_zz, in_x, cross):
    """
    Seed-averaged statistics from a list of sign matrices.
    Returns dict of scalars and the per-seed arrays.
    """
    per_seed = {k: [] for k in STAT_KEYS}
    for s in sign_mats:
        c = alignment_matrix(s)
        st = seed_statistics(c, iu, in_zz, in_x, cross)
        for k in STAT_KEYS:
            per_seed[k].append(st[k])
    per_seed = {k: np.asarray(v, dtype=float)
                for k, v in per_seed.items()}
    means = {k: float(np.nanmean(v))
             for k, v in per_seed.items()}
    return means, per_seed


def bootstrap_ci_over_seeds(per_seed_values, n_boot, rng):
    finite = per_seed_values[np.isfinite(per_seed_values)]
    if finite.size < 2:
        return (np.nan, np.nan)
    m = finite.size
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, m, size=m)
        boot[b] = np.mean(finite[idx])
    return (float(np.percentile(boot, CI_LOW)),
            float(np.percentile(boot, CI_HIGH)))


def permutation_null(sign_mats, iu, in_zz, in_x, cross,
                     n_perm, rng):
    """
    Entrywise independent sign flips per surrogate dataset.
    Zero entries stay zero (flip multiplies by +-1).
    Returns dict: statistic -> null array of length n_perm
    (each entry is the seed-averaged statistic).
    """
    null = {k: np.empty(n_perm, dtype=float)
            for k in STAT_KEYS}
    for p in range(n_perm):
        acc = {k: [] for k in STAT_KEYS}
        for s in sign_mats:
            flips = rng.choice([-1.0, 1.0], size=s.shape)
            c = alignment_matrix(s * flips)
            st = seed_statistics(c, iu, in_zz, in_x, cross)
            for k in STAT_KEYS:
                acc[k].append(st[k])
        for k in STAT_KEYS:
            null[k][p] = float(np.nanmean(acc[k]))
    return null

def run_analysis(zero_tol, n_perm, raw_cache, rng):
    """
    Full Step 5 analysis at one zero threshold.
    Returns (summary DataFrame, zeros DataFrame).
    Signed block statistics receive two-sided null flags
    (above_null, below_null, outside_null); all other
    statistics are upper-tail only.
    """
    summary_rows = []
    zero_rows = []
    for n in N_LIST:
        for d in D_LIST:
            for variant in VARIANTS:
                sign_mats = []
                missing = 0
                for seed in SEEDS:
                    a = raw_cache.get((n, d, variant, seed))
                    if a is None:
                        missing += 1
                    else:
                        sign_mats.append(
                            signs_from_raw(a, zero_tol))
                if missing > 0:
                    warnings.warn(
                        f"n={n}, d={d}, {variant}: "
                        f"{missing} missing npz files."
                    )
                if len(sign_mats) < 2:
                    continue

                m_terms = sign_mats[0].shape[0]
                zz, x = sector_indices(n, m_terms)
                iu, in_zz, in_x, cross = block_masks(
                    m_terms, zz, x
                )

                # Zero fractions per sector
                zf_zz = float(np.mean([
                    np.mean(s[zz, :] == 0)
                    for s in sign_mats
                ]))
                zf_x = float(np.mean([
                    np.mean(s[x, :] == 0)
                    for s in sign_mats
                ]))
                zero_rows.append({
                    "n_qubits": n, "depth": d,
                    "circuit_variant": variant,
                    "zero_fraction_ZZ_terms": zf_zz,
                    "zero_fraction_X_terms": zf_x,
                })

                # Observed statistics
                means, per_seed = dataset_statistics(
                    sign_mats, iu, in_zz, in_x, cross
                )

                # Permutation null
                null = permutation_null(
                    sign_mats, iu, in_zz, in_x, cross,
                    n_perm, rng
                )

                row = {
                    "n_qubits": n, "depth": d,
                    "circuit_variant": variant,
                    "n_seeds_used": len(sign_mats),
                    "n_terms": m_terms,
                }
                for k in STAT_KEYS:
                    lo, hi = bootstrap_ci_over_seeds(
                        per_seed[k], N_BOOT, rng
                    )
                    nlo, nhi = (
                        float(np.percentile(null[k],
                                            CI_LOW)),
                        float(np.percentile(null[k],
                                            CI_HIGH)),
                    )
                    row[f"{k}_obs"] = means[k]
                    row[f"{k}_ci_low"] = lo
                    row[f"{k}_ci_high"] = hi
                    row[f"{k}_null_low"] = nlo
                    row[f"{k}_null_high"] = nhi

                    above = bool(
                        np.isfinite(means[k])
                        and means[k] > nhi
                    )
                    row[f"{k}_above_null"] = above
                    if k in TWO_SIDED_KEYS:
                        below = bool(
                            np.isfinite(means[k])
                            and means[k] < nlo
                        )
                        row[f"{k}_below_null"] = below
                        row[f"{k}_outside_null"] = bool(
                            above or below
                        )

                # Null-relative lam_max metrics
                null_lam_med = float(
                    np.median(null["lam_max"]))
                row["lam_max_null_median"] = null_lam_med
                row["lam_max_excess"] = (
                    means["lam_max"] - null_lam_med
                )
                row["lam_max_ratio"] = (
                    means["lam_max"] / null_lam_med
                    if null_lam_med > 0 else np.nan
                )

                summary_rows.append(row)
                print(f"done: tol={zero_tol:.0e}, "
                      f"n={n}, d={d}, {variant}")
    return (pd.DataFrame(summary_rows),
            pd.DataFrame(zero_rows))

# ------------------------------------------------------------
# Load all raw matrices once (shared across thresholds)
# ------------------------------------------------------------
raw_cache = {}
for n in N_LIST:
    for d in D_LIST:
        for variant in VARIANTS:
            for seed in SEEDS:
                a = load_raw_matrix(n, d, variant, seed)
                if a is not None:
                    raw_cache[(n, d, variant, seed)] = a
print(f"Raw matrices cached: {len(raw_cache)}")

# ------------------------------------------------------------
# Stage A: magnitude distribution at theoretically-zero
# positions (HVA only). See header for the derivation.
# ------------------------------------------------------------
mag_rows = []
for n in N_LIST:
    for d in D_LIST:
        theo_vals = []
        other_vals = []
        for seed in SEEDS:
            a = raw_cache.get((n, d, "hva", seed))
            if a is None:
                continue
            zz_rows = np.arange(0, n - 1)
            theo_vals.append(np.abs(a[zz_rows, -1]))
            mask = np.ones(a.shape, dtype=bool)
            mask[zz_rows, -1] = False
            other_vals.append(np.abs(a[mask]))
        if not theo_vals:
            continue
        theo = np.concatenate(theo_vals)
        other = np.concatenate(other_vals)
        row = {
            "n_qubits": n, "depth": d,
            "theo_p50": float(np.percentile(theo, 50)),
            "theo_p99": float(np.percentile(theo, 99)),
            "theo_max": float(np.max(theo)),
            "other_p01": float(np.percentile(other, 1)),
            "other_p05": float(np.percentile(other, 5)),
            "other_min": float(np.min(other)),
        }
        for tol in ZERO_TOL_LIST:
            row[f"theo_frac_below_{tol:.0e}"] = float(
                np.mean(theo < tol))
            row[f"other_frac_below_{tol:.0e}"] = float(
                np.mean(other < tol))
        mag_rows.append(row)

mag_df = pd.DataFrame(mag_rows)
mag_df.to_csv("hva_theoretical_zero_magnitudes.csv",
              index=False)
print("\nHVA theoretical-zero magnitude check:")
print(mag_df.to_string(index=False))

# ------------------------------------------------------------
# Threshold sweep at N_PERM permutations
# ------------------------------------------------------------
all_summaries = {}
for i, tol in enumerate(ZERO_TOL_LIST):
    # Fresh rng per threshold so each run is reproducible
    # independently of loop order.
    rng_t = np.random.default_rng(BOOT_SEED + i)
    summary_t, zeros_t = run_analysis(
        tol, N_PERM, raw_cache, rng_t)
    tag = f"{tol:.0e}"
    summary_t = summary_t.sort_values(
        ["depth", "circuit_variant", "n_qubits"])
    zeros_t = zeros_t.sort_values(
        ["depth", "circuit_variant", "n_qubits"])
    summary_t.to_csv(
        f"sign_correlation_summary_tol{tag}.csv",
        index=False)
    zeros_t.to_csv(
        f"sign_zero_fractions_tol{tag}.csv",
        index=False)
    all_summaries[tag] = (summary_t, zeros_t)
    print(f"=== threshold {tag} complete ===")

# ------------------------------------------------------------
# Stability comparison across thresholds (key statistics)
# ------------------------------------------------------------
KEY_STATS = ["mean_abs_offdiag_obs", "block_zz_obs",
             "block_x_obs", "block_cross_obs",
             "block_cross_abs_obs", "lam_max_ratio"]
comp_rows = []
for tag, (summary_t, zeros_t) in all_summaries.items():
    merged = summary_t.merge(
        zeros_t,
        on=["n_qubits", "depth", "circuit_variant"])
    for _, r in merged.iterrows():
        row = {
            "zero_tol": tag,
            "n_qubits": r["n_qubits"],
            "depth": r["depth"],
            "circuit_variant": r["circuit_variant"],
            "zero_fraction_ZZ_terms":
                r["zero_fraction_ZZ_terms"],
        }
        for k in KEY_STATS:
            row[k] = r[k]
        comp_rows.append(row)

comp = pd.DataFrame(comp_rows)
comp = comp.sort_values(
    ["circuit_variant", "depth", "n_qubits", "zero_tol"])
comp.to_csv("threshold_stability_comparison.csv",
            index=False)

hva_view = comp[comp["circuit_variant"] == "hva"]
print("\nHVA threshold stability (key statistics):")
print(hva_view.to_string(index=False))

# ------------------------------------------------------------
# Figures at the final threshold
# ------------------------------------------------------------
summary = all_summaries[FINAL_TOL_TAG][0]
depths = sorted(summary["depth"].unique())
style = {
    "hea": {"color": "tab:blue", "marker": "s",
            "label": "HEA"},
    "hva": {"color": "tab:orange", "marker": "o",
            "label": "HVA"},
}


def plot_stat(stat, ylabel, fname):
    fig, axes = plt.subplots(
        1, len(depths),
        figsize=(5.0 * len(depths), 4.0), sharey=True
    )
    if len(depths) == 1:
        axes = [axes]
    for ax, d in zip(axes, depths):
        sub_d = summary[summary["depth"] == d]
        for variant in VARIANTS:
            sub = sub_d[
                sub_d["circuit_variant"] == variant
            ].sort_values("n_qubits")
            if len(sub) == 0:
                continue
            st = style[variant]
            y = sub[f"{stat}_obs"].to_numpy(float)
            lo = sub[f"{stat}_ci_low"].to_numpy(float)
            hi = sub[f"{stat}_ci_high"].to_numpy(float)
            yerr = np.vstack([np.maximum(y - lo, 0),
                              np.maximum(hi - y, 0)])
            ax.errorbar(
                sub["n_qubits"], y, yerr=yerr,
                color=st["color"], marker=st["marker"],
                markersize=7, linewidth=2.0, capsize=4,
                label=st["label"]
            )
            ax.fill_between(
                sub["n_qubits"],
                sub[f"{stat}_null_low"].to_numpy(float),
                sub[f"{stat}_null_high"].to_numpy(float),
                color=st["color"], alpha=0.12,
                label=f"{st['label']} null band"
            )
        ax.set_xlabel("Number of qubits", fontsize=12)
        ax.set_xticks(sorted(summary["n_qubits"].unique()))
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"depth = {d}", fontsize=12)
    axes[0].set_ylabel(ylabel, fontsize=12)
    h, l = axes[-1].get_legend_handles_labels()
    uniq = dict(zip(l, h))
    axes[-1].legend(uniq.values(), uniq.keys(),
                    frameon=True, fontsize=9)
    plt.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.show()


plot_stat("mean_abs_offdiag",
          "Mean absolute off-diagonal alignment",
          "C_mean_abs_offdiag.pdf")
plot_stat("lam_max",
          "Leading eigenvalue of C",
          "C_spectral.pdf")

# Blockwise figure: signed block means per variant
fig, axes = plt.subplots(
    2, len(depths),
    figsize=(5.0 * len(depths), 7.6),
    sharex="col", sharey="row"
)
axes = np.atleast_2d(axes)
block_colors = {
    "block_zz": ("tab:red", "ZZ-ZZ"),
    "block_x": ("tab:green", "X-X"),
    "block_cross": ("tab:gray", "ZZ-X"),
}
for r, variant in enumerate(VARIANTS):
    for c, d in enumerate(depths):
        ax = axes[r, c]
        sub = summary[
            (summary["depth"] == d)
            & (summary["circuit_variant"] == variant)
        ].sort_values("n_qubits")
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        for stat, (color, label) in block_colors.items():
            y = sub[f"{stat}_obs"].to_numpy(float)
            lo = sub[f"{stat}_ci_low"].to_numpy(float)
            hi = sub[f"{stat}_ci_high"].to_numpy(float)
            yerr = np.vstack([np.maximum(y - lo, 0),
                              np.maximum(hi - y, 0)])
            ax.errorbar(
                sub["n_qubits"], y, yerr=yerr,
                color=color, marker="o", markersize=6,
                linewidth=1.8, capsize=3, label=label
            )
        ax.axhline(0.0, color="black", linewidth=0.8,
                   linestyle=":")
        ax.set_xticks(sorted(summary["n_qubits"].unique()))
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, alpha=0.3)
        if r == 0:
            ax.set_title(f"depth = {d}", fontsize=12)
        if r == 1:
            ax.set_xlabel("Number of qubits", fontsize=12)
        if c == 0:
            ax.set_ylabel(
                f"{variant.upper()}\nblock mean alignment",
                fontsize=11
            )
h, l = axes[0, -1].get_legend_handles_labels()
uniq = dict(zip(l, h))
axes[0, -1].legend(uniq.values(), uniq.keys(),
                   frameon=True, fontsize=9)
plt.tight_layout()
fig.savefig("C_blockwise.pdf", bbox_inches="tight")
plt.show()
