# ============================================================
# Step 5 (LFIM version): Sign-correlation structure from raw
# termwise data, 3-sector blocks, ZERO_TOL sweep,
# N_PERM = 1000 permutation null.
#
# Estimand (same as TFIM):
#   For each seed s, the within-circuit, parameter-averaged
#   sign-alignment matrix
#     C^(s)_{ab} = (1/K) sum_k sgn(a_{a,k}) sgn(a_{b,k})
#   The 30 seed-level matrices are the independent replicates.
#
# LFIM structure (verified against
# bp_hva_vs_hea_lfim_save_per_run_v1.py):
#   Terms: M = 3n - 1, ordered as
#     indices 0..n-2      ZZ edges   (coeff -1.0)
#     indices n-1..2n-2   X fields   (coeff hx)
#     indices 2n-1..3n-2  Z fields   (coeff hz)
#   HVA parameters: per layer [beta_x, gamma_zz, beta_z],
#   K = 3d total. The last RZZ block and the last RZ block are
#   both Z-diagonal, so the LAST TWO parameters (columns -2,
#   -1) have exactly zero derivatives against every diagonal
#   term (ZZ and Z sectors). Theoretical zero fraction in the
#   ZZ and Z sectors: 2/(3d); X sector: 0.
#
# Zero handling:
#   Entries with |a| < zero_tol get sign 0. The analysis runs
#   at every threshold in ZERO_TOL_LIST. Stage A checks the
#   magnitude distribution at the theoretically-zero positions
#   to justify the final choice (TFIM-established default is
#   1e-10, to be confirmed here).
#
# Aggregate statistics (per seed -> mean over seeds):
#   1. mean absolute off-diagonal alignment
#   2. blockwise signed AND absolute means over the six
#      sector blocks: ZZ-ZZ, X-X, Z-Z, ZZ-X, ZZ-Z, X-Z
#      (absolute means guard against within-block cancellation
#      of mixed +/- organization)
#   3. leading eigenvalue of C^(s), reported also as excess
#      and ratio relative to the permutation-null median
#
# Uncertainty and null:
#   - Observed CI: bootstrap over the 30 per-seed statistics.
#   - Permutation null: entrywise independent Rademacher sign
#     flips, preserving magnitudes and the zero pattern. This
#     realizes a conditional iid random-sign null: it removes
#     all term-term alignment AND any per-term marginal sign
#     imbalance. Comparison against this null is a CONSISTENCY
#     TEST with the random-sign model, not a pure test of
#     pairwise independence.
#   - Signed block statistics get two-sided flags (above /
#     below / outside); nonnegative-excess statistics keep the
#     upper-tail flag only.
#
# Outputs:
#   lfim_hva_theoretical_zero_magnitudes.csv
#   lfim_sign_correlation_summary_tol{tag}.csv  (per threshold)
#   lfim_sign_zero_fractions_tol{tag}.csv       (per threshold)
#   lfim_threshold_stability_comparison.csv
#   lfim_C_mean_abs_offdiag.pdf
#   lfim_C_blockwise.pdf
#   lfim_C_spectral.pdf
# ============================================================
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
RAW_DIR = "../data/raw_runs_lfim"

N_LIST = [4, 6, 8, 10]
D_LIST = [4, 6, 8]
VARIANTS = ["hea", "hva"]
SEEDS = list(range(30))
NPZ_KEY = "grad_terms"

# Full sweep. If runtime is a concern, [1e-12, 1e-10] suffices
# once Stage A shows a clean gap (TFIM precedent).
ZERO_TOL_LIST = [1e-12, 1e-10, 1e-9, 1e-8]
N_PERM = 1000
N_BOOT = 2000
BOOT_SEED = 20260813
CI_LOW = 2.5
CI_HIGH = 97.5

# Threshold used for the final figures. TFIM-established
# default; confirm with Stage A and the stability table.
FINAL_TOL_TAG = "1e-10"

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
    LFIM term ordering (verified against the experiment
    script): ZZ 0..n-2, X n-1..2n-2, Z 2n-1..3n-2.
    """
    if m_terms != 3 * n - 1:
        raise ValueError(
            f"n={n}: expected {3*n-1} terms, got {m_terms}. "
            "Raw file format does not match the LFIM "
            "3-sector assumption; aborting."
        )
    zz = np.arange(0, n - 1)
    x = np.arange(n - 1, 2 * n - 1)
    z = np.arange(2 * n - 1, m_terms)
    return zz, x, z


def alignment_matrix(s):
    """
    C = (1/K) S S^T from the sign matrix S (M x K).
    """
    k = s.shape[1]
    return (s @ s.T) / float(k)


def block_masks(m_terms, zz, x, z):
    """
    Boolean masks over the strict upper triangle for the six
    sector blocks of the 3-sector LFIM structure.
    """
    iu = np.triu_indices(m_terms, k=1)
    sec = np.zeros(m_terms, dtype=int)
    sec[zz] = 0
    sec[x] = 1
    sec[z] = 2
    s_i = sec[iu[0]]
    s_j = sec[iu[1]]
    masks = {
        "zz": (s_i == 0) & (s_j == 0),
        "x": (s_i == 1) & (s_j == 1),
        "z": (s_i == 2) & (s_j == 2),
        "zz_x": ((s_i == 0) & (s_j == 1))
                | ((s_i == 1) & (s_j == 0)),
        "zz_z": ((s_i == 0) & (s_j == 2))
                | ((s_i == 2) & (s_j == 0)),
        "x_z": ((s_i == 1) & (s_j == 2))
               | ((s_i == 2) & (s_j == 1)),
    }
    return iu, masks


BLOCK_NAMES = ["zz", "x", "z", "zz_x", "zz_z", "x_z"]

STAT_KEYS = (["mean_abs_offdiag"]
             + [f"block_{b}" for b in BLOCK_NAMES]
             + [f"block_{b}_abs" for b in BLOCK_NAMES]
             + ["lam_max"])

# Signed block statistics are physically meaningful in BOTH
# directions (negative values indicate anti-alignment), so
# they get two-sided null flags. All other statistics are
# nonnegative-excess quantities (upper-tail flag only).
TWO_SIDED_KEYS = [f"block_{b}" for b in BLOCK_NAMES]


def seed_statistics(c, iu, masks):
    """
    Aggregate statistics from one seed-level matrix C.
    """
    off = c[iu]

    def m(v):
        return float(np.mean(v)) if v.size > 0 else np.nan

    out = {
        "mean_abs_offdiag": float(np.mean(np.abs(off))),
        "lam_max": float(np.linalg.eigvalsh(c)[-1]),
    }
    for b in BLOCK_NAMES:
        sel = off[masks[b]]
        out[f"block_{b}"] = m(sel)
        out[f"block_{b}_abs"] = m(np.abs(sel))
    return out


def dataset_statistics(sign_mats, iu, masks):
    """
    Seed-averaged statistics from a list of sign matrices.
    """
    per_seed = {k: [] for k in STAT_KEYS}
    for s in sign_mats:
        c = alignment_matrix(s)
        st = seed_statistics(c, iu, masks)
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


def permutation_null(sign_mats, iu, masks, n_perm, rng):
    """
    Entrywise independent sign flips per surrogate dataset.
    Zero entries stay zero (flip multiplies by +-1).
    """
    null = {k: np.empty(n_perm, dtype=float)
            for k in STAT_KEYS}
    for p in range(n_perm):
        acc = {k: [] for k in STAT_KEYS}
        for s in sign_mats:
            flips = rng.choice([-1.0, 1.0], size=s.shape)
            c = alignment_matrix(s * flips)
            st = seed_statistics(c, iu, masks)
            for k in STAT_KEYS:
                acc[k].append(st[k])
        for k in STAT_KEYS:
            null[k][p] = float(np.nanmean(acc[k]))
    return null


def run_analysis(zero_tol, n_perm, raw_cache, rng):
    """
    Full Step 5 analysis at one zero threshold.
    Returns (summary DataFrame, zeros DataFrame).
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
                zz, x, z = sector_indices(n, m_terms)
                iu, masks = block_masks(m_terms, zz, x, z)

                # Zero fractions per sector
                zf_zz = float(np.mean([
                    np.mean(s[zz, :] == 0)
                    for s in sign_mats
                ]))
                zf_x = float(np.mean([
                    np.mean(s[x, :] == 0)
                    for s in sign_mats
                ]))
                zf_z = float(np.mean([
                    np.mean(s[z, :] == 0)
                    for s in sign_mats
                ]))
                zero_rows.append({
                    "n_qubits": n, "depth": d,
                    "circuit_variant": variant,
                    "zero_fraction_ZZ_terms": zf_zz,
                    "zero_fraction_X_terms": zf_x,
                    "zero_fraction_Z_terms": zf_z,
                })

                # Observed statistics
                means, per_seed = dataset_statistics(
                    sign_mats, iu, masks
                )

                # Permutation null
                null = permutation_null(
                    sign_mats, iu, masks, n_perm, rng
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
# positions (LFIM HVA). The last gamma_zz and the last beta_z
# (columns -2 and -1) commute with every diagonal term, so
# their derivatives against ZZ and Z sectors are exactly
# zero analytically. Expected zero fraction: 2/(3d) in the
# ZZ and Z sectors, 0 in the X sector.
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
            diag_rows = np.concatenate([
                np.arange(0, n - 1),
                np.arange(2 * n - 1, 3 * n - 1),
            ])
            last_cols = [a.shape[1] - 2, a.shape[1] - 1]
            theo_vals.append(
                np.abs(a[np.ix_(diag_rows,
                                last_cols)]).ravel())
            mask = np.ones(a.shape, dtype=bool)
            mask[np.ix_(diag_rows, last_cols)] = False
            other_vals.append(np.abs(a[mask]))
        if not theo_vals:
            continue
        theo = np.concatenate(theo_vals)
        other = np.concatenate(other_vals)
        row = {
            "n_qubits": n, "depth": d,
            "expected_zero_frac_2_over_3d": 2.0 / (3.0 * d),
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
mag_df.to_csv("lfim_hva_theoretical_zero_magnitudes.csv",
              index=False)
print("\nLFIM HVA theoretical-zero magnitude check:")
print(mag_df.to_string(index=False))

# ------------------------------------------------------------
# Threshold sweep at N_PERM permutations
# ------------------------------------------------------------
all_summaries = {}
for i, tol in enumerate(ZERO_TOL_LIST):
    # Fresh rng per threshold for order-independent
    # reproducibility.
    rng_t = np.random.default_rng(BOOT_SEED + i)
    summary_t, zeros_t = run_analysis(
        tol, N_PERM, raw_cache, rng_t)
    tag = f"{tol:.0e}"
    summary_t = summary_t.sort_values(
        ["depth", "circuit_variant", "n_qubits"])
    zeros_t = zeros_t.sort_values(
        ["depth", "circuit_variant", "n_qubits"])
    summary_t.to_csv(
        f"lfim_sign_correlation_summary_tol{tag}.csv",
        index=False)
    zeros_t.to_csv(
        f"lfim_sign_zero_fractions_tol{tag}.csv",
        index=False)
    all_summaries[tag] = (summary_t, zeros_t)
    print(f"=== threshold {tag} complete ===")

# ------------------------------------------------------------
# Stability comparison across thresholds (key statistics)
# ------------------------------------------------------------
KEY_STATS = (["mean_abs_offdiag_obs"]
             + [f"block_{b}_obs" for b in BLOCK_NAMES]
             + [f"block_{b}_abs_obs" for b in BLOCK_NAMES]
             + ["lam_max_ratio"])

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
            "zero_fraction_Z_terms":
                r["zero_fraction_Z_terms"],
        }
        for k in KEY_STATS:
            row[k] = r[k]
        comp_rows.append(row)

comp = pd.DataFrame(comp_rows)
comp = comp.sort_values(
    ["circuit_variant", "depth", "n_qubits", "zero_tol"])
comp.to_csv("lfim_threshold_stability_comparison.csv",
            index=False)

hva_view = comp[comp["circuit_variant"] == "hva"]
print("\nLFIM HVA threshold stability (key statistics):")
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
          "lfim_C_mean_abs_offdiag.pdf")
plot_stat("lam_max",
          "Leading eigenvalue of C",
          "lfim_C_spectral.pdf")

# Blockwise figure: signed block means per variant,
# six blocks (within-sector solid, cross-sector dashed)
fig, axes = plt.subplots(
    2, len(depths),
    figsize=(5.2 * len(depths), 7.8),
    sharex="col", sharey="row"
)
axes = np.atleast_2d(axes)
block_style = {
    "block_zz": ("tab:red", "-", "ZZ-ZZ"),
    "block_x": ("tab:green", "-", "X-X"),
    "block_z": ("tab:blue", "-", "Z-Z"),
    "block_zz_x": ("tab:gray", "--", "ZZ-X"),
    "block_zz_z": ("tab:purple", "--", "ZZ-Z"),
    "block_x_z": ("tab:olive", "--", "X-Z"),
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
        for stat, (color, ls, label) in block_style.items():
            y = sub[f"{stat}_obs"].to_numpy(float)
            lo = sub[f"{stat}_ci_low"].to_numpy(float)
            hi = sub[f"{stat}_ci_high"].to_numpy(float)
            yerr = np.vstack([np.maximum(y - lo, 0),
                              np.maximum(hi - y, 0)])
            ax.errorbar(
                sub["n_qubits"], y, yerr=yerr,
                color=color, linestyle=ls, marker="o",
                markersize=5, linewidth=1.6, capsize=3,
                label=label
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
                   frameon=True, fontsize=8, ncol=2)
plt.tight_layout()
fig.savefig("lfim_C_blockwise.pdf", bbox_inches="tight")
plt.show()
