#!/usr/bin/env python3
# bp_hva_vs_hea_save_per_run_v4.py
#
# Compare HEA vs HVA on TFIM using destructive-interference diagnostics.
#
# Changes from v3
#  - FD sensitivity check for HVA 
#
# Example:
#   python bp_hva_vs_hea_save_per_run_v4.py \
#       --n_qubits 4,6 \
#       --depths 4,6 \
#       --seeds 10 \
#       --outdir_root ./runs \
#       --save_raw_terms false
#
# Main outputs:
# - per_run.csv
# - grad_mean_check.csv
# - layer_summary.csv
# - structure_compare.csv
# - summary.json
# - figures/*.pdf
#
# Ansatz definitions:
# - HEA:
#     each layer = trainable RY-RZ on every qubit, then entangler ring
# - HVA:
#     each layer = exp(-i beta sum X_i) followed by exp(-i gamma sum Z_i Z_{i+1})
#   implemented exactly using commuting gate products:
#     RX(2*beta) on each qubit, then RZZ(2*gamma) on each edge

import os
import csv
import json
import time
import argparse
from itertools import product

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_N_QUBITS = "4,6,8,10"
DEFAULT_DEPTHS = "4,6,8"
DEFAULT_SEEDS = 10
DEFAULT_OUTDIR_ROOT = "./runs"
DEFAULT_SAVE_RAW_TERMS = False
DEFAULT_FD_EPS = 1e-5

H_FIELD = 1.0
RNG_INIT_LOW = -np.pi
RNG_INIT_HIGH = np.pi


def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_bool(s):
    s = str(s).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError("Invalid bool: " + str(s))


def make_timestamped_outdir(root):
    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(root, "bp_hva_vs_hea_" + ts)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "figures"), exist_ok=True)
    return outdir


def rx_mat(t):
    c = np.cos(t / 2.0)
    s = np.sin(t / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def ry_mat(t):
    c = np.cos(t / 2.0)
    s = np.sin(t / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz_mat(t):
    return np.diag([
        np.exp(-1j * t / 2.0),
        np.exp(1j * t / 2.0),
    ]).astype(np.complex128)


def apply_gate_sv(psi, gate, qubit, n):
    shape = [2] * n
    psi_t = psi.reshape(shape)
    ax = n - 1 - qubit
    psi_t = np.tensordot(gate, psi_t, axes=([1], [ax]))
    psi_t = np.moveaxis(psi_t, 0, ax)
    return psi_t.reshape(2 ** n)


_CNOT_CACHE = {}
_RZZ_CACHE = {}


def get_cnot(ctrl, tgt, n):
    key = (ctrl, tgt, n)
    if key in _CNOT_CACHE:
        return _CNOT_CACHE[key]
    dim = 2 ** n
    U = np.zeros((dim, dim), dtype=np.complex128)
    for k in range(dim):
        if (k >> ctrl) & 1:
            U[k ^ (1 << tgt), k] = 1.0
        else:
            U[k, k] = 1.0
    _CNOT_CACHE[key] = U
    return U


def get_rzz(theta, q1, q2, n):
    key = (round(float(theta), 12), q1, q2, n)
    if key in _RZZ_CACHE:
        return _RZZ_CACHE[key]
    dim = 2 ** n
    U = np.zeros((dim, dim), dtype=np.complex128)
    for k in range(dim):
        b1 = (k >> q1) & 1
        b2 = (k >> q2) & 1
        z1 = 1.0 if b1 == 0 else -1.0
        z2 = 1.0 if b2 == 0 else -1.0
        phase = np.exp(-1j * theta * z1 * z2 / 2.0)
        U[k, k] = phase
    _RZZ_CACHE[key] = U
    return U


def build_h_obc_matrix(n, h=1.0):
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    def kron_all(ops):
        r = ops[0]
        for o in ops[1:]:
            r = np.kron(r, o)
        return r

    for i in range(n - 1):
        ops = [I2] * n
        ops[i] = Z
        ops[i + 1] = Z
        H -= kron_all(ops)

    for i in range(n):
        ops = [I2] * n
        ops[i] = X
        H += h * kron_all(ops)

    return H


def pauli_terms_for_h_obc(n, h=1.0):
    terms = []
    for i in range(n - 1):
        terms.append(("zz", i, i + 1, -1.0))
    for i in range(n):
        terms.append(("x", i, None, h))
    return terms


def build_h_term_matrix(n, term):
    kind, i, j, coeff = term
    I2 = np.eye(2, dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    ops = [I2] * n
    if kind == "zz":
        ops[i] = Z
        ops[j] = Z
    elif kind == "x":
        ops[i] = X
    else:
        raise ValueError("Unknown term kind: " + str(kind))

    M = ops[0]
    for o in ops[1:]:
        M = np.kron(M, o)
    return float(coeff), M


def build_h_term_matrices(n, h=1.0):
    mats = []
    for term in pauli_terms_for_h_obc(n, h):
        coeff, M = build_h_term_matrix(n, term)
        mats.append((term, coeff, M))
    return mats


def n_params_for_variant(n, depth, variant):
    if variant == "hea":
        return 2 * n * depth
    if variant == "hva":
        return 2 * depth
    raise ValueError("Unknown variant: " + str(variant))


def final_state_sv(params, n, depth, variant):
    dim = 2 ** n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    idx = 0

    if variant == "hea":
        for _layer in range(depth):
            for q in range(n):
                psi = apply_gate_sv(psi, ry_mat(params[idx]), q, n)
                idx += 1
                psi = apply_gate_sv(psi, rz_mat(params[idx]), q, n)
                idx += 1
            for q in range(n - 1):
                psi = get_cnot(q, q + 1, n) @ psi
            if n > 1:
                psi = get_cnot(n - 1, 0, n) @ psi
        return psi

    if variant == "hva":
        for _layer in range(depth):
            beta = params[idx]
            idx += 1
            gamma = params[idx]
            idx += 1

            for q in range(n):
                psi = apply_gate_sv(psi, rx_mat(2.0 * beta), q, n)

            for q in range(n - 1):
                psi = get_rzz(2.0 * gamma, q, q + 1, n) @ psi
        return psi

    raise ValueError("Unknown variant: " + str(variant))


def sim_sv(params, n, depth, H_mat, variant):
    psi = final_state_sv(params, n, depth, variant)
    e = np.real(np.vdot(psi, H_mat @ psi))
    return float(e)

def grad_sv(params, n, depth, H_mat, variant, fd_eps=1e-5):
    npar = len(params)
    g = np.zeros(npar, dtype=np.float64)

    if variant == "hea":
        shift = np.pi / 2.0
        pref = 0.5

        for k in range(npar):
            pp = params.copy()
            pm = params.copy()
            pp[k] += shift
            pm[k] -= shift
            ep = sim_sv(pp, n, depth, H_mat, variant)
            em = sim_sv(pm, n, depth, H_mat, variant)
            g[k] = pref * (ep - em)

    elif variant == "hva":
        for k in range(npar):
            pp = params.copy()
            pm = params.copy()
            pp[k] += fd_eps
            pm[k] -= fd_eps
            ep = sim_sv(pp, n, depth, H_mat, variant)
            em = sim_sv(pm, n, depth, H_mat, variant)
            g[k] = (ep - em) / (2.0 * fd_eps)

    else:
        raise ValueError("Unknown variant: " + str(variant))

    return g


def grad_termwise_sv(params, n, depth, term_mats, variant, fd_eps=1e-5):
    npar = len(params)
    nterms = len(term_mats)
    out = np.zeros((npar, nterms), dtype=np.float64)

    if variant == "hea":
        shift = np.pi / 2.0
        pref = 0.5

        for k in range(npar):
            pp = params.copy()
            pm = params.copy()
            pp[k] += shift
            pm[k] -= shift

            psi_p = final_state_sv(pp, n, depth, variant)
            psi_m = final_state_sv(pm, n, depth, variant)

            for t_idx, (_, h_coeff, M) in enumerate(term_mats):
                ep = np.real(np.vdot(psi_p, M @ psi_p))
                em = np.real(np.vdot(psi_m, M @ psi_m))
                out[k, t_idx] = h_coeff * pref * (ep - em)

    elif variant == "hva":
        for k in range(npar):
            pp = params.copy()
            pm = params.copy()
            pp[k] += fd_eps
            pm[k] -= fd_eps

            psi_p = final_state_sv(pp, n, depth, variant)
            psi_m = final_state_sv(pm, n, depth, variant)

            for t_idx, (_, h_coeff, M) in enumerate(term_mats):
                ep = np.real(np.vdot(psi_p, M @ psi_p))
                em = np.real(np.vdot(psi_m, M @ psi_m))
                out[k, t_idx] = h_coeff * (ep - em) / (2.0 * fd_eps)

    else:
        raise ValueError("Unknown variant: " + str(variant))

    return out

def signed_cancellation_ratio(a):
    denom = float(np.sum(np.abs(a)))
    numer = float(np.abs(np.sum(a)))
    if denom <= 1e-15:
        return 0.0
    return numer / denom


def effective_term_count(a):
    abs_a = np.abs(a)
    denom = float(np.sum(abs_a ** 2))
    if denom <= 1e-15:
        return 0.0
    numer = float(np.sum(abs_a) ** 2)
    return numer / denom


def inv_sqrt_safe(x):
    if x <= 1e-15:
        return 0.0
    return 1.0 / np.sqrt(x)


def sign_vector(a, zero_tol=1e-12):
    s = np.zeros_like(a, dtype=np.int8)
    s[a > zero_tol] = 1
    s[a < -zero_tol] = -1
    return s


def sign_agreement_ratio(s1, s2):
    return float(np.mean(s1 == s2))


def sign_correlation(s1, s2):
    x = s1.astype(np.float64)
    y = s2.astype(np.float64)
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx <= 1e-15 or ny <= 1e-15:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))


def param_index_to_layer_and_type(k, n, variant):
    if variant == "hea":
        layer = k // (2 * n)
        offset = k % (2 * n)
        qubit = offset // 2
        gate_type = "ry" if (offset % 2 == 0) else "rz"
        return layer, qubit, gate_type

    if variant == "hva":
        layer = k // 2
        gate_type = "beta_x" if (k % 2 == 0) else "gamma_zz"
        return layer, -1, gate_type

    raise ValueError("Unknown variant: " + str(variant))


def summarize_layer_locality(values_k, n, depth, variant):
    vals = []
    per_layer = 2 * n if variant == "hea" else 2
    for layer in range(depth):
        start = layer * per_layer
        end = (layer + 1) * per_layer
        vals.append(float(np.mean(values_k[start:end])))
    return vals

def run_single(n, depth, variant, seed, H_mat, term_mats, fd_eps=DEFAULT_FD_EPS):
    rng = np.random.RandomState(seed)
    npar = n_params_for_variant(n, depth, variant)
    params = rng.uniform(RNG_INIT_LOW, RNG_INIT_HIGH, size=npar).astype(np.float64)

    grad_full = grad_sv(params, n, depth, H_mat, variant, fd_eps)
    grad_terms = grad_termwise_sv(params, n, depth, term_mats, variant, fd_eps)

    recon = np.sum(grad_terms, axis=1)
    recon_abs_err = np.abs(recon - grad_full)
    max_err = float(np.max(recon_abs_err))
    mean_err = float(np.mean(recon_abs_err))
    if max_err > 1e-8:
        print(
            "[WARN] reconstruction mismatch ({0}): max_err={1:.3e}, mean_err={2:.3e}".format(
                variant, max_err, mean_err
            )
        )

    R_k = np.zeros(npar, dtype=np.float64)
    N_eff_k = np.zeros(npar, dtype=np.float64)
    inv_sqrt_N_eff_k = np.zeros(npar, dtype=np.float64)
    B_eff_k = np.zeros(npar, dtype=np.float64)
    abs_sum_k = np.zeros(npar, dtype=np.float64)
    cancel_gap_k = np.zeros(npar, dtype=np.float64)
    pos_frac_k = np.zeros(npar, dtype=np.float64)
    neg_frac_k = np.zeros(npar, dtype=np.float64)
    sign_patterns = np.zeros_like(grad_terms, dtype=np.int8)
    Q_k = np.zeros(npar, dtype=np.float64)

    for k in range(npar):
        a = grad_terms[k]
        R_k[k] = signed_cancellation_ratio(a)
        N_eff_k[k] = effective_term_count(a)
        inv_sqrt_N_eff_k[k] = inv_sqrt_safe(N_eff_k[k])
        B_eff_k[k] = float(R_k[k] * np.sqrt(N_eff_k[k])) if N_eff_k[k] > 1e-15 else 0.0
        abs_sum_k[k] = float(np.sum(np.abs(a)))
        cancel_gap_k[k] = float(np.sum(np.abs(a)) - np.abs(np.sum(a)))
        sign_patterns[k] = sign_vector(a)
        pos_frac_k[k] = float(np.mean(a > 1e-12))
        neg_frac_k[k] = float(np.mean(a < -1e-12))
        Q_k[k] = float(np.sum(a ** 2))

    if np.std(inv_sqrt_N_eff_k) > 1e-15 and np.std(R_k) > 1e-15:
        corr_R_vs_inv = float(np.corrcoef(inv_sqrt_N_eff_k, R_k)[0, 1])
    else:
        corr_R_vs_inv = 0.0

    var_bridge_actual = float(np.mean((B_eff_k ** 2) * Q_k))
    var_bridge_approx = float(np.mean(B_eff_k ** 2) * np.mean(Q_k))
    if var_bridge_approx > 1e-15:
        var_bridge_ratio = float(var_bridge_actual / var_bridge_approx)
    else:
        var_bridge_ratio = 0.0

    return {
        "n_qubits": n,
        "depth": depth,
        "circuit_variant": variant,
        "seed": seed,
        "n_params": npar,
        "grad_full": grad_full,
        "grad_norm_mean": float(np.mean(np.abs(grad_full))),
        "grad_norm_l2": float(np.linalg.norm(grad_full)),
        "R_k": R_k,
        "R_mean": float(np.mean(R_k)),
        "R_by_layer": summarize_layer_locality(R_k, n, depth, variant),
        "N_eff_k": N_eff_k,
        "N_eff_mean": float(np.mean(N_eff_k)),
        "N_eff_by_layer": summarize_layer_locality(N_eff_k, n, depth, variant),
        "inv_sqrt_N_eff_k": inv_sqrt_N_eff_k,
        "B_eff_k": B_eff_k,
        "B_eff_mean_param": float(np.mean(B_eff_k)),
        "B_eff_std_param": float(np.std(B_eff_k)),
        "Q_k": Q_k,
        "Q_mean": float(np.mean(Q_k)),
        "corr_R_vs_inv_sqrt_Neff": corr_R_vs_inv,
        "var_bridge_actual": var_bridge_actual,
        "var_bridge_approx": var_bridge_approx,
        "var_bridge_ratio": var_bridge_ratio,
        "abs_sum_k": abs_sum_k,
        "cancel_gap_k": cancel_gap_k,
        "pos_frac_k": pos_frac_k,
        "neg_frac_k": neg_frac_k,
        "sign_patterns": sign_patterns,
        "grad_terms": grad_terms,
        "recon_max_abs_error": max_err,
        "recon_mean_abs_error": mean_err,
    }

def compute_sign_stability(run_group):
    n_runs = len(run_group)
    npar = run_group[0]["sign_patterns"].shape[0]

    pairwise_by_param = np.zeros((n_runs, n_runs, npar), dtype=np.float64)
    corr_by_param = np.zeros((n_runs, n_runs, npar), dtype=np.float64)

    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            s1 = run_group[i]["sign_patterns"]
            s2 = run_group[j]["sign_patterns"]
            for k in range(npar):
                pairwise_by_param[i, j, k] = sign_agreement_ratio(s1[k], s2[k])
                corr_by_param[i, j, k] = sign_correlation(s1[k], s2[k])

    iu = np.triu_indices(n_runs, k=1)
    vals = pairwise_by_param[iu[0], iu[1], :].ravel()
    corr_vals = corr_by_param[iu[0], iu[1], :].ravel()

    mean_agree = float(np.mean(vals)) if vals.size > 0 else 0.0
    mean_corr = float(np.mean(corr_vals)) if corr_vals.size > 0 else 0.0

    return {
        "sign_stability_mean": mean_agree,
        "sign_corr_mean": mean_corr,
    }


def compute_structure_sensitivity(all_runs):
    grouped = {}
    for r in all_runs:
        key = (r["n_qubits"], r["depth"], r["circuit_variant"])
        grouped.setdefault(key, []).append(r)

    rows = []
    by_nd = {}
    for (n, d, variant), lst in grouped.items():
        by_nd.setdefault((n, d), {})[variant] = lst

    for (n, d), variant_dict in sorted(by_nd.items()):
        variants = sorted(variant_dict.keys())
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                v1 = variants[i]
                v2 = variants[j]
                g1 = variant_dict[v1]
                g2 = variant_dict[v2]

                R1 = float(np.mean([x["R_mean"] for x in g1]))
                R2 = float(np.mean([x["R_mean"] for x in g2]))
                N1 = float(np.mean([x["N_eff_mean"] for x in g1]))
                N2 = float(np.mean([x["N_eff_mean"] for x in g2]))
                B1 = float(np.mean([x["B_eff_mean_param"] for x in g1]))
                B2 = float(np.mean([x["B_eff_mean_param"] for x in g2]))

                common_seeds = sorted(set(x["seed"] for x in g1) & set(x["seed"] for x in g2))
                corr_vals = []
                for seed in common_seeds:
                    a = [x for x in g1 if x["seed"] == seed][0]
                    b = [x for x in g2 if x["seed"] == seed][0]
                    npar = min(a["sign_patterns"].shape[0], b["sign_patterns"].shape[0])
                    tmp = []
                    for k in range(npar):
                        tmp.append(sign_correlation(a["sign_patterns"][k], b["sign_patterns"][k]))
                    corr_vals.append(float(np.mean(tmp)))

                rows.append({
                    "n_qubits": n,
                    "depth": d,
                    "variant_a": v1,
                    "variant_b": v2,
                    "R_mean_a": R1,
                    "R_mean_b": R2,
                    "R_diff": R2 - R1,
                    "R_diff_abs": abs(R1 - R2),
                    "N_eff_mean_a": N1,
                    "N_eff_mean_b": N2,
                    "N_eff_diff": N2 - N1,
                    "N_eff_diff_abs": abs(N1 - N2),
                    "B_eff_mean_a": B1,
                    "B_eff_mean_b": B2,
                    "B_eff_diff": B2 - B1,
                    "sign_pattern_corr_mean": float(np.mean(corr_vals)) if len(corr_vals) > 0 else 0.0,
                })
    return rows


def bootstrap_ci_of_mean_diff(x, y, n_boot=5000, seed=123):
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    delta = y - x
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boots[i] = float(np.mean(delta[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(np.mean(delta)), float(lo), float(hi)


def classify_case(delta_R, delta_B, eps_R=1e-3, eps_B=5e-2):
    if delta_R > eps_R and delta_B > eps_B:
        return "support"
    if abs(delta_R) <= eps_R and abs(delta_B) <= eps_B:
        return "null"
    return "fail"


def save_config(outdir, cfg):
    path = os.path.join(outdir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print("Saved:", path)


def save_per_run_csv(outdir, all_runs):
    path = os.path.join(outdir, "per_run.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "run_id",
            "n_qubits",
            "depth",
            "circuit_variant",
            "seed",
            "n_params",
            "param_index",
            "layer_index",
            "qubit_index",
            "gate_type",
            "R_k",
            "N_eff_k",
            "inv_sqrt_N_eff_k",
            "B_eff_k",
            "grad_k",
            "abs_sum_k",
            "cancel_gap_k",
            "pos_frac_k",
            "neg_frac_k",
        ])
        for r in all_runs:
            n = r["n_qubits"]
            depth = r["depth"]
            variant = r["circuit_variant"]
            seed = r["seed"]
            run_id = "n{0}_d{1}_{2}_seed{3:02d}".format(n, depth, variant, seed)
            npar = len(r["R_k"])
            grad_total = np.sum(r["grad_terms"], axis=1)
            for k in range(npar):
                layer, qubit, gate_type = param_index_to_layer_and_type(k, n, variant)
                w.writerow([
                    run_id,
                    n,
                    depth,
                    variant,
                    seed,
                    r["n_params"],
                    k,
                    layer,
                    qubit,
                    gate_type,
                    float(r["R_k"][k]),
                    float(r["N_eff_k"][k]),
                    float(r["inv_sqrt_N_eff_k"][k]),
                    float(r["B_eff_k"][k]),
                    float(grad_total[k]),
                    float(r["abs_sum_k"][k]),
                    float(r["cancel_gap_k"][k]),
                    float(r["pos_frac_k"][k]),
                    float(r["neg_frac_k"][k]),
                ])
    print("Saved:", path)


def save_layer_summary_csv(outdir, all_runs):
    path = os.path.join(outdir, "layer_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "run_id",
            "n_qubits",
            "depth",
            "circuit_variant",
            "seed",
            "n_params",
            "layer",
            "R_layer",
            "N_eff_layer",
        ])
        for r in all_runs:
            n = r["n_qubits"]
            depth = r["depth"]
            variant = r["circuit_variant"]
            seed = r["seed"]
            run_id = "n{0}_d{1}_{2}_seed{3:02d}".format(n, depth, variant, seed)
            for layer in range(len(r["R_by_layer"])):
                w.writerow([
                    run_id,
                    n,
                    depth,
                    variant,
                    seed,
                    r["n_params"],
                    layer,
                    float(r["R_by_layer"][layer]),
                    float(r["N_eff_by_layer"][layer]),
                ])
    print("Saved:", path)


def save_structure_compare_csv(outdir, rows):
    path = os.path.join(outdir, "structure_compare.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "n_qubits",
            "depth",
            "variant_a",
            "variant_b",
            "R_mean_a",
            "R_mean_b",
            "R_diff",
            "R_diff_abs",
            "N_eff_mean_a",
            "N_eff_mean_b",
            "N_eff_diff",
            "N_eff_diff_abs",
            "B_eff_mean_a",
            "B_eff_mean_b",
            "B_eff_diff",
            "sign_pattern_corr_mean",
        ])
        for row in rows:
            w.writerow([
                row["n_qubits"],
                row["depth"],
                row["variant_a"],
                row["variant_b"],
                row["R_mean_a"],
                row["R_mean_b"],
                row["R_diff"],
                row["R_diff_abs"],
                row["N_eff_mean_a"],
                row["N_eff_mean_b"],
                row["N_eff_diff"],
                row["N_eff_diff_abs"],
                row["B_eff_mean_a"],
                row["B_eff_mean_b"],
                row["B_eff_diff"],
                row["sign_pattern_corr_mean"],
            ])
    print("Saved:", path)


def save_raw_terms(outdir, all_runs):
    data = {}
    for r in all_runs:
        key = "n{0}_d{1}_{2}_seed{3:02d}".format(
            r["n_qubits"], r["depth"], r["circuit_variant"], r["seed"]
        )
        data[key + "_grad_terms"] = r["grad_terms"]
        data[key + "_sign_patterns"] = r["sign_patterns"]
        data[key + "_N_eff_k"] = r["N_eff_k"]
        data[key + "_R_k"] = r["R_k"]
        data[key + "_B_eff_k"] = r["B_eff_k"]
    path = os.path.join(outdir, "raw_terms.npz")
    np.savez_compressed(path, **data)
    print("Saved:", path)


def build_summary(all_runs):
    grouped = {}
    for r in all_runs:
        key = (r["n_qubits"], r["depth"], r["circuit_variant"])
        grouped.setdefault(key, []).append(r)

    per_group = []
    scatter_rows = []

    for key in sorted(grouped.keys()):
        lst = grouped[key]
        n, depth, variant = key
        sign_stats = compute_sign_stability(lst)

        grad_mat = np.stack([x["grad_full"] for x in lst], axis=0)
        mean_grad_k = np.mean(grad_mat, axis=0)
        abs_mean_grad_k = np.abs(mean_grad_k)
        mean_abs_grad_k = np.mean(np.abs(grad_mat), axis=0)
        std_grad_k = np.std(grad_mat, axis=0)
        bias_ratio_k = abs_mean_grad_k / (mean_abs_grad_k + 1e-15)

        all_R = np.concatenate([x["R_k"] for x in lst], axis=0)
        all_inv = np.concatenate([x["inv_sqrt_N_eff_k"] for x in lst], axis=0)
        all_B = np.concatenate([x["B_eff_k"] for x in lst], axis=0)
        all_Q = np.concatenate([x["Q_k"] for x in lst], axis=0)

        if np.std(all_R) > 1e-15 and np.std(all_inv) > 1e-15:
            corr_group = float(np.corrcoef(all_inv, all_R)[0, 1])
        else:
            corr_group = 0.0

        if np.std(all_B ** 2) > 1e-15 and np.std(all_Q) > 1e-15:
            corr_b2_q = float(np.corrcoef(all_B ** 2, all_Q)[0, 1])
        else:
            corr_b2_q = 0.0

        group_var_bridge_actual = float(np.mean([x["var_bridge_actual"] for x in lst]))
        group_var_bridge_approx = float(np.mean([x["var_bridge_approx"] for x in lst]))
        group_var_bridge_ratio = (
            float(group_var_bridge_actual / group_var_bridge_approx)
            if group_var_bridge_approx > 1e-15 else 0.0
        )

        per_group.append({
            "n_qubits": n,
            "depth": depth,
            "circuit_variant": variant,
            "n_params_mean": float(np.mean([x["n_params"] for x in lst])),
            "R_mean_over_seeds": float(np.mean([x["R_mean"] for x in lst])),
            "R_std_over_seeds": float(np.std([x["R_mean"] for x in lst])),
            "N_eff_mean_over_seeds": float(np.mean([x["N_eff_mean"] for x in lst])),
            "N_eff_std_over_seeds": float(np.std([x["N_eff_mean"] for x in lst])),
            "B_eff_mean_over_seeds": float(np.mean([x["B_eff_mean_param"] for x in lst])),
            "B_eff_std_over_seeds": float(np.std([x["B_eff_mean_param"] for x in lst])),
            "B_eff_param_mean_all": float(np.mean(all_B)),
            "B_eff_param_std_all": float(np.std(all_B)),
            "grad_norm_l2_mean": float(np.mean([x["grad_norm_l2"] for x in lst])),
            "sign_stability_mean": sign_stats["sign_stability_mean"],
            "sign_corr_mean": sign_stats["sign_corr_mean"],
            "R_by_layer_mean": np.mean(
                np.array([x["R_by_layer"] for x in lst], dtype=np.float64), axis=0
            ).tolist(),
            "N_eff_by_layer_mean": np.mean(
                np.array([x["N_eff_by_layer"] for x in lst], dtype=np.float64), axis=0
            ).tolist(),
            "corr_R_vs_inv_sqrt_Neff": corr_group,
            "Q_mean_over_seeds": float(np.mean([x["Q_mean"] for x in lst])),
            "Q_std_over_seeds": float(np.std([x["Q_mean"] for x in lst])),
            "var_bridge_actual": group_var_bridge_actual,
            "var_bridge_approx": group_var_bridge_approx,
            "var_bridge_ratio": group_var_bridge_ratio,
            "corr_B2_vs_Q": corr_b2_q,
            "mean_abs_param_mean_grad": float(np.mean(abs_mean_grad_k)),
            "max_abs_param_mean_grad": float(np.max(abs_mean_grad_k)),
            "mean_bias_ratio": float(np.mean(bias_ratio_k)),
            "max_bias_ratio": float(np.max(bias_ratio_k)),
            "recon_max_abs_error_max": float(np.max([x["recon_max_abs_error"] for x in lst])),
            "recon_mean_abs_error_mean": float(np.mean([x["recon_mean_abs_error"] for x in lst])),
        })

        for run in lst:
            run_id = "n{0}_d{1}_{2}_seed{3:02d}".format(
                n, depth, variant, run["seed"]
            )
            for rk, invk, bk in zip(run["R_k"], run["inv_sqrt_N_eff_k"], run["B_eff_k"]):
                scatter_rows.append({
                    "run_id": run_id,
                    "n_qubits": n,
                    "depth": depth,
                    "circuit_variant": variant,
                    "R_k": float(rk),
                    "inv_sqrt_N_eff_k": float(invk),
                    "B_eff_k": float(bk),
                })

    structure_compare = compute_structure_sensitivity(all_runs)

    by_group = {}
    for r in all_runs:
        key = (r["n_qubits"], r["depth"])
        if key not in by_group:
            by_group[key] = {}
        if r["circuit_variant"] not in by_group[key]:
            by_group[key][r["circuit_variant"]] = []
        by_group[key][r["circuit_variant"]].append(r)

    paired_summary = []
    for key in sorted(by_group.keys()):
        n, depth = key
        variants = by_group[key]
        if "hea" not in variants or "hva" not in variants:
            continue

        runs_a = sorted(variants["hea"], key=lambda x: x["seed"])
        runs_b = sorted(variants["hva"], key=lambda x: x["seed"])
        common_seeds = sorted(set(x["seed"] for x in runs_a) & set(x["seed"] for x in runs_b))

        a_R = []
        b_R = []
        a_N = []
        b_N = []
        a_B = []
        b_B = []

        for seed in common_seeds:
            a = [x for x in runs_a if x["seed"] == seed][0]
            b = [x for x in runs_b if x["seed"] == seed][0]
            a_R.append(a["R_mean"])
            b_R.append(b["R_mean"])
            a_N.append(a["N_eff_mean"])
            b_N.append(b["N_eff_mean"])
            a_B.append(a["B_eff_mean_param"])
            b_B.append(b["B_eff_mean_param"])

        dR_mean, dR_lo, dR_hi = bootstrap_ci_of_mean_diff(a_R, b_R)
        dN_mean, dN_lo, dN_hi = bootstrap_ci_of_mean_diff(a_N, b_N)
        dB_mean, dB_lo, dB_hi = bootstrap_ci_of_mean_diff(a_B, b_B)
        case = classify_case(dR_mean, dB_mean)

        paired_summary.append({
            "n_qubits": n,
            "depth": depth,
            "variant_a": "hea",
            "variant_b": "hva",
            "n_common_seeds": len(common_seeds),
            "hea_R_mean": float(np.mean(a_R)),
            "hva_R_mean": float(np.mean(b_R)),
            "delta_R_mean": dR_mean,
            "delta_R_ci95": [dR_lo, dR_hi],
            "hea_N_eff_mean": float(np.mean(a_N)),
            "hva_N_eff_mean": float(np.mean(b_N)),
            "delta_N_eff_mean": dN_mean,
            "delta_N_eff_ci95": [dN_lo, dN_hi],
            "hea_B_eff_mean": float(np.mean(a_B)),
            "hva_B_eff_mean": float(np.mean(b_B)),
            "delta_B_eff_mean": dB_mean,
            "delta_B_eff_ci95": [dB_lo, dB_hi],
            "case_label": case,
        })

    return {
        "per_group": per_group,
        "structure_compare": structure_compare,
        "R_vs_inv_sqrt_Neff_scatter": scatter_rows,
        "paired_summary": paired_summary,
    }


def save_summary(outdir, summary):
    path = os.path.join(outdir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", path)

def save_grad_mean_check_csv(outdir, all_runs):
    path = os.path.join(outdir, "grad_mean_check.csv")

    grouped = {}
    for r in all_runs:
        key = (r["n_qubits"], r["depth"], r["circuit_variant"])
        grouped.setdefault(key, []).append(r)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "n_qubits",
            "depth",
            "circuit_variant",
            "param_index",
            "layer_index",
            "qubit_index",
            "gate_type",
            "grad_mean_over_seeds",
            "abs_grad_mean_over_seeds",
            "grad_abs_mean_over_seeds",
            "grad_std_over_seeds",
            "bias_ratio",
        ])

        for key in sorted(grouped.keys()):
            lst = grouped[key]
            n, depth, variant = key
            grad_mat = np.stack([x["grad_full"] for x in lst], axis=0)

            mean_grad_k = np.mean(grad_mat, axis=0)
            abs_mean_grad_k = np.abs(mean_grad_k)
            mean_abs_grad_k = np.mean(np.abs(grad_mat), axis=0)
            std_grad_k = np.std(grad_mat, axis=0)
            bias_ratio_k = abs_mean_grad_k / (mean_abs_grad_k + 1e-15)

            npar = grad_mat.shape[1]
            for k in range(npar):
                layer, qubit, gate_type = param_index_to_layer_and_type(k, n, variant)
                w.writerow([
                    n,
                    depth,
                    variant,
                    k,
                    layer,
                    qubit,
                    gate_type,
                    float(mean_grad_k[k]),
                    float(abs_mean_grad_k[k]),
                    float(mean_abs_grad_k[k]),
                    float(std_grad_k[k]),
                    float(bias_ratio_k[k]),
                ])

    print("Saved:", path)


def plot_R_vs_depth(outdir, summary):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    groups = summary["per_group"]
    nq_values = sorted(set(g["n_qubits"] for g in groups))
    variants = sorted(set(g["circuit_variant"] for g in groups))

    for n in nq_values:
        for variant in variants:
            sub = [g for g in groups if g["n_qubits"] == n and g["circuit_variant"] == variant]
            if len(sub) == 0:
                continue
            sub = sorted(sub, key=lambda x: x["depth"])
            x = [g["depth"] for g in sub]
            y = [g["R_mean_over_seeds"] for g in sub]
            ax.plot(x, y, marker="o", label="n={0}, {1}".format(n, variant))

    ax.set_xlabel("Depth")
    ax.set_ylabel("R_mean")
    ax.set_title("HEA vs HVA: cancellation ratio vs depth")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    path = os.path.join(outdir, "figures", "R_vs_depth.pdf")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def plot_Beff_vs_depth(outdir, summary):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    groups = summary["per_group"]
    nq_values = sorted(set(g["n_qubits"] for g in groups))
    variants = sorted(set(g["circuit_variant"] for g in groups))

    for n in nq_values:
        for variant in variants:
            sub = [g for g in groups if g["n_qubits"] == n and g["circuit_variant"] == variant]
            if len(sub) == 0:
                continue
            sub = sorted(sub, key=lambda x: x["depth"])
            x = [g["depth"] for g in sub]
            y = [g["B_eff_mean_over_seeds"] for g in sub]
            ax.plot(x, y, marker="o", label="n={0}, {1}".format(n, variant))

    ax.set_xlabel("Depth")
    ax.set_ylabel("B_eff_mean")
    ax.set_title("HEA vs HVA: B_eff vs depth")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    path = os.path.join(outdir, "figures", "B_eff_vs_depth.pdf")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def plot_Neff_vs_depth(outdir, summary):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    groups = summary["per_group"]
    nq_values = sorted(set(g["n_qubits"] for g in groups))
    variants = sorted(set(g["circuit_variant"] for g in groups))

    for n in nq_values:
        for variant in variants:
            sub = [g for g in groups if g["n_qubits"] == n and g["circuit_variant"] == variant]
            if len(sub) == 0:
                continue
            sub = sorted(sub, key=lambda x: x["depth"])
            x = [g["depth"] for g in sub]
            y = [g["N_eff_mean_over_seeds"] for g in sub]
            ax.plot(x, y, marker="o", label="n={0}, {1}".format(n, variant))

    ax.set_xlabel("Depth")
    ax.set_ylabel("N_eff_mean")
    ax.set_title("HEA vs HVA: N_eff vs depth")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    path = os.path.join(outdir, "figures", "N_eff_vs_depth.pdf")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def plot_R_vs_inv_sqrt_Neff(outdir, summary):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    rows = summary["R_vs_inv_sqrt_Neff_scatter"]
    by_group = {}

    for row in rows:
        key = (row["n_qubits"], row["depth"], row["circuit_variant"])
        if key not in by_group:
            by_group[key] = {"x": [], "y": []}
        by_group[key]["x"].append(row["inv_sqrt_N_eff_k"])
        by_group[key]["y"].append(row["R_k"])

    for key in sorted(by_group.keys()):
        n, depth, variant = key
        x = np.array(by_group[key]["x"], dtype=np.float64)
        y = np.array(by_group[key]["y"], dtype=np.float64)
        ax.scatter(x, y, s=14, alpha=0.55, label="n={0}, d={1}, {2}".format(n, depth, variant))

    ref = np.linspace(0.0, 1.0, 200)
    ax.plot(ref, ref, linestyle="--", linewidth=1.2, label="R = 1/sqrt(N_eff)")
    ax.set_xlabel("1 / sqrt(N_eff)")
    ax.set_ylabel("R_k")
    ax.set_title("HEA vs HVA: R_k vs 1/sqrt(N_eff)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    path = os.path.join(outdir, "figures", "R_vs_inv_sqrt_Neff.pdf")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def plot_delta_bar(outdir, summary):
    groups = summary["per_group"]
    hea = {}
    hva = {}

    for g in groups:
        key = (g["n_qubits"], g["depth"])
        if g["circuit_variant"] == "hea":
            hea[key] = g
        elif g["circuit_variant"] == "hva":
            hva[key] = g

    keys = sorted(set(hea.keys()) & set(hva.keys()))
    labels = []
    delta_R = []
    delta_N = []
    delta_B = []

    for key in keys:
        n, depth = key
        labels.append("n={0},d={1}".format(n, depth))
        delta_R.append(hva[key]["R_mean_over_seeds"] - hea[key]["R_mean_over_seeds"])
        delta_N.append(hva[key]["N_eff_mean_over_seeds"] - hea[key]["N_eff_mean_over_seeds"])
        delta_B.append(hva[key]["B_eff_mean_over_seeds"] - hea[key]["B_eff_mean_over_seeds"])

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width, delta_R, width, label="Delta R")
    ax.bar(x, delta_N, width, label="Delta N_eff")
    ax.bar(x + width, delta_B, width, label="Delta B_eff")
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("HVA - HEA")
    ax.set_title("HEA vs HVA deltas")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    path = os.path.join(outdir, "figures", "hea_vs_hva_delta_bar.pdf")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def save_raw_run(outdir, run):
    key = "n{0}_d{1}_{2}_seed{3:02d}".format(
        run["n_qubits"], run["depth"],
        run["circuit_variant"], run["seed"]
    )
    path = os.path.join(outdir, "raw_runs", key + "_grad_terms.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, grad_terms=run["grad_terms"])

def main():
    parser = argparse.ArgumentParser(
        description="Compare HEA vs HVA on TFIM using destructive-interference diagnostics."
    )
    parser.add_argument("--n_qubits", type=str, default=DEFAULT_N_QUBITS)
    parser.add_argument("--depths", type=str, default=DEFAULT_DEPTHS)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--outdir_root", type=str, default=DEFAULT_OUTDIR_ROOT)
    parser.add_argument("--save_raw_terms", type=str, default=str(DEFAULT_SAVE_RAW_TERMS).lower())
    parser.add_argument("--fd_eps", type=float, default=DEFAULT_FD_EPS)
    args = parser.parse_args()

    n_qubits_list = parse_int_list(args.n_qubits)
    depths = parse_int_list(args.depths)
    seeds = int(args.seeds)
    save_raw = parse_bool(args.save_raw_terms)
    fd_eps = float(args.fd_eps)

    outdir = make_timestamped_outdir(args.outdir_root)

    cfg = {
        "n_qubits": n_qubits_list,
        "depths": depths,
        "seeds": seeds,
        "outdir_root": args.outdir_root,
        "save_raw_terms": save_raw,
        "hamiltonian": "TFIM_OBC",
        "h_field": H_FIELD,
        "fd_eps": fd_eps,
        "variants": ["hea", "hva"],
        "hea_definition": "layerwise RY-RZ on each qubit plus ring entangler",
        "hva_definition": "RX field block plus nearest-neighbor RZZ problem block",
        "pre_registered_hypotheses": {
            "H1": "HVA should have larger mean R than HEA",
            "H3": "HVA should have larger mean B_eff than HEA",
            "H4": "HVA should show better depth robustness than HEA",
            "Mechanism": "N_eff is interpreted after observing B_eff differences",
        },
        "timestamp_outdir": outdir,
    }
    save_config(outdir, cfg)

    t0 = time.time()
    all_runs = []

    h_cache = {}
    term_cache = {}
    variants = ["hea", "hva"]

    total_jobs = len(n_qubits_list) * len(depths) * len(variants) * seeds
    done = 0

    for n, depth, variant, seed in product(n_qubits_list, depths, variants, range(seeds)):
        if n not in h_cache:
            h_cache[n] = build_h_obc_matrix(n, h=H_FIELD)
            term_cache[n] = build_h_term_matrices(n, h=H_FIELD)

        run = run_single(
            n=n,
            depth=depth,
            variant=variant,
            seed=seed,
            H_mat=h_cache[n],
            term_mats=term_cache[n],
            fd_eps=fd_eps,
        )
        all_runs.append(run)

        if save_raw:
            save_raw_run(outdir, run)

        done += 1
        print(
            "[{0}/{1}] n={2} depth={3} variant={4} seed={5} n_params={6} "
            "R_mean={7:.4f} N_eff_mean={8:.4f} B_eff_mean={9:.4f}".format(
                done,
                total_jobs,
                n,
                depth,
                variant,
                seed,
                run["n_params"],
                run["R_mean"],
                run["N_eff_mean"],
                run["B_eff_mean_param"],
            )
        )

    summary = build_summary(all_runs)

    save_summary(outdir, summary)
    save_per_run_csv(outdir, all_runs)
    save_grad_mean_check_csv(outdir, all_runs)
    save_layer_summary_csv(outdir, all_runs)
    save_structure_compare_csv(outdir, summary["structure_compare"])

    # if save_raw:
    #     save_raw_terms(outdir, all_runs)

    plot_R_vs_depth(outdir, summary)
    plot_Beff_vs_depth(outdir, summary)
    plot_Neff_vs_depth(outdir, summary)
    plot_R_vs_inv_sqrt_Neff(outdir, summary)
    plot_delta_bar(outdir, summary)

    dt = time.time() - t0
    print("=" * 72)
    print("DONE: {0:.1f}s ({1:.2f}h)".format(dt, dt / 3600.0))
    print("Results in: {0}".format(outdir))
    print("=" * 72)


if __name__ == "__main__":
    main()
