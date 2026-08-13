# ============================================================
# Step 3: Exact second-moment bridge decomposition
#
# Quantities (row level):
#   Q_k   = abs_sum_k^2 / N_eff_k
#   B2_k  = B_eff_k^2
#   B2Q_k = B2_k * Q_k          (equals grad_k^2 exactly)
#
# Group level (parameter-first estimator throughout):
#   M2  = E[B2Q]                     (exact second moment E[g^2])
#   F   = E[B2Q] / (E[B2] E[Q])     (factorization ratio)
#   MGR = E_param[(mean_s g)^2] / E_param[mean_s g^2]
#         (mean-gradient correction ratio)
#   param_avg_var_plugin = M2 * (1 - MGR)  (plug-in, ddof=0)
#
# Decomposition across consecutive n (algebraic identity):
#   dlog M2 = dlog E[B2] + dlog E[Q] + dlog F
#   Shares are shares of the SECOND-MOMENT log change,
#   not causal attribution of variance.
#
# Uncertainty:
#   (a) Group quantities: joint bootstrap over seeds.
#   (b) M2 and F log-linear slopes: bootstrap with independent
#       seed resampling within each n condition.
#   (c) dlog components per adjacent n-pair: joint bootstrap
#       with independent seed resampling in the two conditions.
#   Shares are reported as point estimates only (no CI), since
#   the denominator dlog M2 can approach zero.
#
# Outputs:
#   lfim_bridge_decomposition_summary.csv
#   lfim_bridge_decomposition_steps.csv
#   lfim_m2_f_scaling_fits.csv
#   lfim_second_moment_decomposition.pdf
#   lfim_f_ratio_vs_n.pdf
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
N_BOOT = 5000
BOOT_SEED = 20260813
CI_LOW = 2.5
CI_HIGH = 97.5

rng = np.random.default_rng(BOOT_SEED)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def make_param_seed_frame(data, value_col):
    """
    Parameter x seed DataFrame for one (n, d, variant) group.
    """
    frame = data.pivot_table(
        index="param_index",
        columns="seed",
        values=value_col,
        aggfunc="mean"
    )
    return frame.sort_index(axis=0).sort_index(axis=1)


def build_aligned_matrices(sub):
    """
    Build Q, B2, B2Q, g matrices with defensively aligned
    rows (param_index) and columns (seed).
    Returns four float ndarrays with identical shape.
    """
    frames = {
        "Q": make_param_seed_frame(sub, "Q_k"),
        "B2": make_param_seed_frame(sub, "B2_k"),
        "B2Q": make_param_seed_frame(sub, "B2Q_k"),
        "g": make_param_seed_frame(sub, "grad_k"),
    }
    common_rows = frames["Q"].index
    common_cols = frames["Q"].columns
    for f in frames.values():
        common_rows = common_rows.intersection(f.index)
        common_cols = common_cols.intersection(f.columns)
    out = {}
    for k, f in frames.items():
        out[k] = f.loc[common_rows, common_cols].to_numpy(float)
    return out["Q"], out["B2"], out["B2Q"], out["g"]


def group_point_estimates(mq, mb2, mb2q, mg):
    """
    All group-level point quantities from parameter x seed
    matrices (parameter-first estimator).
    """
    e_q = float(np.nanmean(np.nanmean(mq, axis=1)))
    e_b2 = float(np.nanmean(np.nanmean(mb2, axis=1)))
    e_m2 = float(np.nanmean(np.nanmean(mb2q, axis=1)))

    if e_q > 0 and e_b2 > 0:
        f_ratio = e_m2 / (e_b2 * e_q)
    else:
        f_ratio = np.nan

    mean_g_per_param = np.nanmean(mg, axis=1)
    mean_g2_per_param = np.nanmean(mg ** 2, axis=1)
    num = float(np.nanmean(mean_g_per_param ** 2))
    den = float(np.nanmean(mean_g2_per_param))
    mgr = num / den if den > 0 else np.nan

    # Empirical plug-in variance (ddof=0), parameter-averaged.
    var_per_param = mean_g2_per_param - mean_g_per_param ** 2
    param_avg_var_plugin = float(np.nanmean(var_per_param))

    return {
        "E_Q": e_q,
        "E_B2": e_b2,
        "M2": e_m2,
        "F": f_ratio,
        "mean_grad_ratio": mgr,
        "param_avg_var_plugin": param_avg_var_plugin,
    }


def joint_bootstrap_group(mq, mb2, mb2q, mg, n_boot, rng):
    """
    Joint bootstrap over seeds within one group.
    """
    n_seeds = mq.shape[1]
    keys = ["E_Q", "E_B2", "M2", "F", "mean_grad_ratio"]
    if n_seeds < 2:
        empty = np.array([], dtype=float)
        return {k: empty for k in keys}
    out = {k: np.empty(n_boot, dtype=float) for k in keys}
    for b in range(n_boot):
        cols = rng.integers(0, n_seeds, size=n_seeds)
        est = group_point_estimates(
            mq[:, cols], mb2[:, cols],
            mb2q[:, cols], mg[:, cols]
        )
        for k in keys:
            out[k][b] = est[k]
    return out


def bootstrap_slopes_m2_f(mats_per_n, n_boot, rng):
    """
    Bootstrap log-linear slopes of M2 and F vs n.
    Seeds are resampled independently within each n condition.
    mats_per_n: {n: (mq, mb2, mb2q, mg)}
    """
    n_values = sorted(mats_per_n.keys())
    slopes_m2 = []
    slopes_f = []
    for _ in range(n_boot):
        x = []
        y_m2 = []
        y_f = []
        valid = True
        for n in n_values:
            mq, mb2, mb2q, mg = mats_per_n[n]
            n_seeds = mq.shape[1]
            cols = rng.integers(0, n_seeds, size=n_seeds)
            est = group_point_estimates(
                mq[:, cols], mb2[:, cols],
                mb2q[:, cols], mg[:, cols]
            )
            if (not np.isfinite(est["M2"]) or est["M2"] <= 0
                    or not np.isfinite(est["F"])
                    or est["F"] <= 0):
                valid = False
                break
            x.append(float(n))
            y_m2.append(np.log(est["M2"]))
            y_f.append(np.log(est["F"]))
        if valid and len(x) >= 2:
            s_m2, _ = np.polyfit(np.asarray(x),
                                 np.asarray(y_m2), 1)
            s_f, _ = np.polyfit(np.asarray(x),
                                np.asarray(y_f), 1)
            slopes_m2.append(float(s_m2))
            slopes_f.append(float(s_f))
    return (np.asarray(slopes_m2, dtype=float),
            np.asarray(slopes_f, dtype=float))


def bootstrap_pair_dlog(mats_a, mats_b, n_boot, rng):
    """
    Joint bootstrap of dlog components for one adjacent
    n-pair. The two conditions come from independent runs,
    so their seeds are resampled independently, but all four
    dlog components are computed on the same replicate.
    """
    keys = ["dlog_M2", "dlog_E_B2", "dlog_E_Q", "dlog_F"]
    out = {k: [] for k in keys}
    mq_a, mb2_a, mb2q_a, mg_a = mats_a
    mq_b, mb2_b, mb2q_b, mg_b = mats_b
    ns_a = mq_a.shape[1]
    ns_b = mq_b.shape[1]
    if ns_a < 2 or ns_b < 2:
        return {k: np.array([], dtype=float) for k in keys}
    for _ in range(n_boot):
        ca = rng.integers(0, ns_a, size=ns_a)
        cb = rng.integers(0, ns_b, size=ns_b)
        ea = group_point_estimates(
            mq_a[:, ca], mb2_a[:, ca],
            mb2q_a[:, ca], mg_a[:, ca]
        )
        eb = group_point_estimates(
            mq_b[:, cb], mb2_b[:, cb],
            mb2q_b[:, cb], mg_b[:, cb]
        )
        vals = [ea["M2"], eb["M2"], ea["E_B2"], eb["E_B2"],
                ea["E_Q"], eb["E_Q"], ea["F"], eb["F"]]
        if any((not np.isfinite(v)) or v <= 0 for v in vals):
            continue
        out["dlog_M2"].append(
            np.log(eb["M2"]) - np.log(ea["M2"]))
        out["dlog_E_B2"].append(
            np.log(eb["E_B2"]) - np.log(ea["E_B2"]))
        out["dlog_E_Q"].append(
            np.log(eb["E_Q"]) - np.log(ea["E_Q"]))
        out["dlog_F"].append(
            np.log(eb["F"]) - np.log(ea["F"]))
    return {k: np.asarray(v, dtype=float)
            for k, v in out.items()}


def pctl_ci(arr):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(finite, CI_LOW)),
            float(np.percentile(finite, CI_HIGH)))


def linear_fit_with_r2(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(intercept), float(r2)

# ------------------------------------------------------------
# Load and validate data
# ------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
required = [
    "n_qubits", "depth", "circuit_variant", "seed",
    "param_index", "grad_k", "B_eff_k", "abs_sum_k",
    "N_eff_k",
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
w_df["B2Q_k"] = w_df["B2_k"] * w_df["Q_k"]

identity_err = (w_df["B2Q_k"] - w_df["grad_k"] ** 2).abs()
print("\nRow-level identity check (B2*Q vs grad^2):")
print(f"  max abs error    = {identity_err.max():.3e}")
print(f"  median abs error = {identity_err.median():.3e}")

key_cols = ["n_qubits", "depth", "circuit_variant",
            "seed", "param_index"]
dup = w_df.duplicated(subset=key_cols, keep=False).sum()
print(f"\nDuplicate parameter-seed rows: {dup}")

seed_cov = (
    w_df.groupby(
        ["n_qubits", "depth", "circuit_variant",
         "param_index"], observed=True
    )["seed"].nunique()
)
bad = (seed_cov != EXPECTED_SEEDS).sum()
if bad > 0:
    warnings.warn(f"{bad} parameter groups lack "
                  f"{EXPECTED_SEEDS} seeds.")

# ------------------------------------------------------------
# Build and cache aligned matrices per group
# ------------------------------------------------------------
group_mats = {}
group_cols = ["n_qubits", "depth", "circuit_variant"]
for keys, sub in w_df.groupby(group_cols, observed=True):
    group_mats[keys] = build_aligned_matrices(sub)

# ------------------------------------------------------------
# Group-level estimates with joint bootstrap
# ------------------------------------------------------------
summary_rows = []
for keys in sorted(group_mats.keys()):
    n_qubits, depth, variant = keys
    mq, mb2, mb2q, mg = group_mats[keys]

    est = group_point_estimates(mq, mb2, mb2q, mg)
    boot = joint_bootstrap_group(mq, mb2, mb2q, mg,
                                 N_BOOT, rng)

    m2_lo, m2_hi = pctl_ci(boot["M2"])
    f_lo, f_hi = pctl_ci(boot["F"])
    mgr_lo, mgr_hi = pctl_ci(boot["mean_grad_ratio"])

    summary_rows.append({
        "n_qubits": n_qubits,
        "depth": depth,
        "circuit_variant": variant,
        "E_Q": est["E_Q"],
        "E_B2": est["E_B2"],
        "M2": est["M2"],
        "ci_low_M2": m2_lo,
        "ci_high_M2": m2_hi,
        "F": est["F"],
        "ci_low_F": f_lo,
        "ci_high_F": f_hi,
        "mean_grad_ratio": est["mean_grad_ratio"],
        "ci_low_mgr": mgr_lo,
        "ci_high_mgr": mgr_hi,
        "param_avg_var_plugin": est["param_avg_var_plugin"],
        "var_plugin_over_M2": (
            est["param_avg_var_plugin"] / est["M2"]
            if est["M2"] > 0 else np.nan
        ),
        "n_params": mq.shape[0],
        "n_unique_seeds": mq.shape[1],
    })

summary = pd.DataFrame(summary_rows)
summary = summary.sort_values(
    ["depth", "circuit_variant", "n_qubits"]
)
summary.to_csv("lfim_bridge_decomposition_summary.csv",
               index=False)
print("\nBridge decomposition summary:")
print(summary.to_string(index=False))

# ------------------------------------------------------------
# Log-linear fits for M2 and F with bootstrap slope CIs
# ------------------------------------------------------------
print("\nLog-linear fits with bootstrap slope CIs:")
fit_rows = []
for (depth, variant), sub_s in summary.groupby(
    ["depth", "circuit_variant"], observed=True
):
    sub_s = sub_s.sort_values("n_qubits")
    x = sub_s["n_qubits"].to_numpy(float)

    mats_per_n = {
        int(n): group_mats[(int(n), depth, variant)]
        for n in x
        if (int(n), depth, variant) in group_mats
    }
    boot_s_m2, boot_s_f = bootstrap_slopes_m2_f(
        mats_per_n, N_BOOT, rng
    )

    for col, label, boot_slopes in [
        ("M2", "M2", boot_s_m2),
        ("F", "F", boot_s_f),
    ]:
        y = sub_s[col].to_numpy(float)
        valid = np.isfinite(y) & (y > 0)
        if valid.sum() < 2:
            continue
        slope, intercept, r2 = linear_fit_with_r2(
            x[valid], np.log(y[valid])
        )
        s_lo, s_hi = pctl_ci(boot_slopes)
        fit_rows.append({
            "depth": depth, "variant": variant,
            "quantity": label,
            "slope": slope,
            "slope_ci_low": s_lo,
            "slope_ci_high": s_hi,
            "factor_per_qubit": float(np.exp(slope)),
            "r2": r2,
        })
        print(f"depth={depth}, variant={variant}, "
              f"{label}: slope={slope:+.4f} "
              f"[{s_lo:+.4f}, {s_hi:+.4f}], "
              f"factor={np.exp(slope):.4f}, R2={r2:.4f}")

pd.DataFrame(fit_rows).to_csv("lfim_m2_f_scaling_fits.csv",
                              index=False)

# ------------------------------------------------------------
# Stepwise decomposition with pair-bootstrap CIs
# ------------------------------------------------------------
step_rows = []
for (depth, variant), sub_s in summary.groupby(
    ["depth", "circuit_variant"], observed=True
):
    sub_s = sub_s.sort_values("n_qubits").reset_index(
        drop=True)
    for i in range(len(sub_s) - 1):
        a = sub_s.iloc[i]
        b = sub_s.iloc[i + 1]
        vals = [a["M2"], b["M2"], a["E_B2"], b["E_B2"],
                a["E_Q"], b["E_Q"], a["F"], b["F"]]
        if any((not np.isfinite(v)) or v <= 0
               for v in vals):
            continue
        d_m2 = np.log(b["M2"]) - np.log(a["M2"])
        d_b2 = np.log(b["E_B2"]) - np.log(a["E_B2"])
        d_q = np.log(b["E_Q"]) - np.log(a["E_Q"])
        d_f = np.log(b["F"]) - np.log(a["F"])
        resid = d_m2 - (d_b2 + d_q + d_f)

        key_a = (int(a["n_qubits"]), depth, variant)
        key_b = (int(b["n_qubits"]), depth, variant)
        boot = bootstrap_pair_dlog(
            group_mats[key_a], group_mats[key_b],
            N_BOOT, rng
        )
        m2_lo, m2_hi = pctl_ci(boot["dlog_M2"])
        b2_lo, b2_hi = pctl_ci(boot["dlog_E_B2"])
        q_lo, q_hi = pctl_ci(boot["dlog_E_Q"])
        f_lo, f_hi = pctl_ci(boot["dlog_F"])

        step_rows.append({
            "depth": depth, "variant": variant,
            "n_from": int(a["n_qubits"]),
            "n_to": int(b["n_qubits"]),
            "dlog_M2": d_m2,
            "ci_low_dlog_M2": m2_lo,
            "ci_high_dlog_M2": m2_hi,
            "dlog_E_B2": d_b2,
            "ci_low_dlog_E_B2": b2_lo,
            "ci_high_dlog_E_B2": b2_hi,
            "dlog_E_Q": d_q,
            "ci_low_dlog_E_Q": q_lo,
            "ci_high_dlog_E_Q": q_hi,
            "dlog_F": d_f,
            "ci_low_dlog_F": f_lo,
            "ci_high_dlog_F": f_hi,
            "identity_residual": resid,
            "share_B2_of_M2_change": (
                d_b2 / d_m2 if d_m2 != 0 else np.nan),
            "share_Q_of_M2_change": (
                d_q / d_m2 if d_m2 != 0 else np.nan),
            "share_F_of_M2_change": (
                d_f / d_m2 if d_m2 != 0 else np.nan),
        })

steps = pd.DataFrame(step_rows)
steps.to_csv("lfim_bridge_decomposition_steps.csv", index=False)
print("\nStepwise decomposition "
      "(shares are of the second-moment log change; "
      "shares reported without CIs):")
print(steps.to_string(index=False))

# ------------------------------------------------------------
# Figure 1: grouped bars with CIs per component
# ------------------------------------------------------------
depths = sorted(summary["depth"].unique())
variants = ["hea", "hva"]
comp_info = [
    ("dlog_E_Q", "tab:green", "dlog E[Q]"),
    ("dlog_E_B2", "tab:purple", "dlog E[B2]"),
    ("dlog_F", "tab:gray", "dlog F"),
]
bar_w = 0.22

fig, axes = plt.subplots(
    len(variants), len(depths),
    figsize=(5.2 * len(depths), 3.9 * len(variants)),
    sharex=False, sharey="row"
)
axes = np.atleast_2d(axes)

for r, variant in enumerate(variants):
    for c, depth in enumerate(depths):
        ax = axes[r, c]
        sub = steps[(steps["depth"] == depth)
                    & (steps["variant"] == variant)]
        sub = sub.sort_values("n_from")
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        xlab = [f"{int(p)}-{int(q)}" for p, q in
                zip(sub["n_from"], sub["n_to"])]
        xpos = np.arange(len(sub))

        for j, (comp, color, label) in enumerate(comp_info):
            vals = sub[comp].to_numpy(float)
            lo = sub[f"ci_low_{comp}"].to_numpy(float)
            hi = sub[f"ci_high_{comp}"].to_numpy(float)
            yerr = np.vstack([
                np.maximum(vals - lo, 0),
                np.maximum(hi - vals, 0),
            ])
            ax.bar(xpos + (j - 1) * bar_w, vals,
                   width=bar_w, color=color, label=label,
                   yerr=yerr, capsize=3,
                   error_kw={"linewidth": 1.0})

        m2 = sub["dlog_M2"].to_numpy(float)
        m2_lo = sub["ci_low_dlog_M2"].to_numpy(float)
        m2_hi = sub["ci_high_dlog_M2"].to_numpy(float)
        ax.errorbar(
            xpos, m2,
            yerr=np.vstack([np.maximum(m2 - m2_lo, 0),
                            np.maximum(m2_hi - m2, 0)]),
            fmt="ko-", markersize=5, linewidth=1.5,
            capsize=3, label="dlog M2 (total)"
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(xpos)
        ax.set_xticklabels(xlab)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        if r == 0:
            ax.set_title(f"depth = {depth}", fontsize=12)
        if c == 0:
            ax.set_ylabel(
                f"{variant.upper()}\n"
                "second-moment log change per n-step",
                fontsize=11
            )

handles, labels = axes[0, 0].get_legend_handles_labels()
uniq = dict(zip(labels, handles))
axes[0, -1].legend(uniq.values(), uniq.keys(),
                   frameon=True, fontsize=9)
plt.tight_layout()
fig.savefig("lfim_second_moment_decomposition.pdf",
            bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# Figure 2: F ratio vs n
# ------------------------------------------------------------
fig2, axes2 = plt.subplots(
    1, len(depths), figsize=(5.0 * len(depths), 4.0),
    sharey=True
)
if len(depths) == 1:
    axes2 = [axes2]
style = {
    "hea": {"color": "tab:blue", "marker": "s",
            "label": "HEA"},
    "hva": {"color": "tab:orange", "marker": "o",
            "label": "HVA"},
}
for ax, depth in zip(axes2, depths):
    sub_d = summary[summary["depth"] == depth]
    for variant in variants:
        sub = sub_d[
            sub_d["circuit_variant"] == variant
        ].sort_values("n_qubits")
        if len(sub) == 0:
            continue
        st = style[variant]
        y = sub["F"].to_numpy(float)
        lo = sub["ci_low_F"].to_numpy(float)
        hi = sub["ci_high_F"].to_numpy(float)
        yerr = np.vstack([np.maximum(y - lo, 0),
                          np.maximum(hi - y, 0)])
        ax.errorbar(sub["n_qubits"], y, yerr=yerr,
                    color=st["color"], marker=st["marker"],
                    markersize=7, linewidth=2.0, capsize=4,
                    label=st["label"])
    ax.axhline(1.0, color="gray", linestyle="--",
               linewidth=1.5, label="F = 1")
    ax.set_xlabel("Number of qubits", fontsize=12)
    ax.set_xticks(sorted(summary["n_qubits"].unique()))
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"depth = {depth}", fontsize=12)
axes2[0].set_ylabel("Factorization ratio F", fontsize=12)
h, l = axes2[-1].get_legend_handles_labels()
uniq = dict(zip(l, h))
axes2[-1].legend(uniq.values(), uniq.keys(),
                 frameon=True, fontsize=10)
plt.tight_layout()
fig2.savefig("lfim_f_ratio_vs_n.pdf", bbox_inches="tight")
plt.show()

# MGR bias-corrected with joint seed bootstrap

rows = []

for keys, sub in w_df.groupby(group_cols, observed=True):
    n_qubits, depth, variant = keys

    g = make_param_seed_frame(
        sub,
        "grad_k"
    ).to_numpy(float)

    S = g.shape[1]

    def mgr_bc_from(gm, bootstrap=False):
        mean_g2 = np.nanmean(
            gm ** 2,
            axis=1
        )

        s2 = np.nanvar(
            gm,
            axis=1,
            ddof=1
        )

        if bootstrap:
            s2 = s2 * S / (S - 1)

        num = np.nanmean(
            mean_g2 - s2
        )

        den = np.nanmean(
            mean_g2
        )

        return (
            num / den
            if den > 0
            else np.nan
        )

    point = mgr_bc_from(
        g,
        bootstrap=False
    )

    boot = np.empty(
        N_BOOT,
        dtype=float
    )

    for b in range(N_BOOT):
        cols = rng.integers(
            0,
            S,
            size=S
        )

        boot[b] = mgr_bc_from(
            g[:, cols],
            bootstrap=True
        )

    lo, hi = pctl_ci(boot)

    rows.append({
        "n_qubits": n_qubits,
        "depth": depth,
        "circuit_variant": variant,
        "MGR_bc": point,
        "ci_low_MGR_bc": lo,
        "ci_high_MGR_bc": hi,
        "one_over_S": 1.0 / S,
    })


mgr_bc_df = pd.DataFrame(
    rows
).sort_values(
    [
        "depth",
        "circuit_variant",
        "n_qubits",
    ]
)

print(
    mgr_bc_df.to_string(
        index=False
    )
)

mgr_bc_df.to_csv(
    "lfim_mgr_bias_corrected.csv",
    index=False
)
