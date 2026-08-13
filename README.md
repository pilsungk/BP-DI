# Barren Plateaus as Destructive Interference

This repository contains the experiment scripts, analysis pipeline, and
result data for the paper:

> Pilsung Kang, "Barren Plateaus as Destructive Interference: A Diagnostic
> Framework and Implications for Structured Ansatzes",
> [arXiv:2605.01319](https://arxiv.org/abs/2605.01319).

The study compares the hardware-efficient ansatz (HEA) and the Hamiltonian
variational ansatz (HVA) on two related Ising models (TFIM and LFIM) using a
termwise diagnostic decomposition of the gradient second moment into an
activity factor (Q), an organization factor (B^2), and their coupling (F),
together with a sign-alignment analysis of the raw termwise gradient
contributions.

## Repository Layout

```
BP-DI/
|-- src/        Experiment drivers (data generation)
|-- analysis/   Step 1-5 analysis scripts (TFIM and LFIM versions)
|-- results/    per_run.csv and all analysis output CSVs
|   |-- tfim/
|   |-- lfim/
|-- data/       Raw termwise gradient matrices (compressed)
```

| Path | Contents |
| --- | --- |
| `src/bp_hva_vs_hea_save_per_run_v4.py` | TFIM experiment (HEA vs HVA) |
| `src/bp_hva_vs_hea_lfim_save_per_run_v1.py` | LFIM experiment (HEA vs HVA) |
| `analysis/step1_q_scaling_{tfim,lfim}.py` | Q (activity) scaling |
| `analysis/step2_b2_scaling_{tfim,lfim}.py` | B^2 (organization) scaling |
| `analysis/step3_decomposition_{tfim,lfim}.py` | Second-moment log-slope decomposition, F, and bias-corrected mean-gradient correction |
| `analysis/step4_coupling_{tfim,lfim}.py` | Within/between-seed decomposition of B^2-Q coupling |
| `analysis/step5_sign_alignment_{tfim,lfim}.py` | Termwise sign-alignment matrices, permutation null, zero-threshold sweep |
| `results/tfim/` | `per_run.csv` + analysis outputs (no prefix) |
| `results/lfim/` | `per_run.csv` + analysis outputs (`lfim_` prefix) |
| `data/raw_runs_tfim.tar.gz` | 720 npz files (termwise gradient matrices, TFIM) |
| `data/raw_runs_lfim.tar.gz` | 720 npz files (termwise gradient matrices, LFIM) |

Figures are not stored in the repository: every figure in the paper is
regenerated deterministically by the analysis scripts.

## Requirements

Python 3 with `numpy`, `pandas`, and `matplotlib`. No quantum-computing
frameworks are required. State evolution is exact NumPy statevector
simulation; gradients are estimated with the parameter-shift rule for the
HEA and central finite differences (`eps = 1e-5`) for the HVA.

## Reproduction

Reproduction is a two-stage pipeline. Stage 1 (data generation) takes hours
and is optional, because its outputs (`per_run.csv` and the raw npz files)
are already included. Stage 2 (analysis) takes minutes.

### Stage 1 (optional): regenerate the data

```
python src/bp_hva_vs_hea_save_per_run_v4.py \
    --n_qubits 4,6,8,10 --depths 4,6,8 --seeds 30 \
    --outdir_root ./runs --save_raw_terms true

python src/bp_hva_vs_hea_lfim_save_per_run_v1.py \
    --n_qubits 4,6,8,10 --depths 4,6,8 --seeds 30 \
    --outdir_root ./runs --save_raw_terms true
```

All runs use fixed random seeds (0-29) for reproducibility.
See the script headers for the Hamiltonians, ansatz definitions, and the
full list of run-level outputs (`per_run.csv`, `layer_summary.csv`,
`structure_compare.csv`, `summary.json`, figures). The TFIM driver
additionally writes `grad_mean_check.csv` (included in `results/tfim/`): 
per-parameter gradient statistics over seeds, including the bias ratio
`|mean(g)| / mean(|g|)` used to check the near-zero-mean assumption
behind the variance bridge.

### Stage 2: run the analysis

Steps 1-4 read `results/{tfim,lfim}/per_run.csv`. Step 5 additionally needs
the raw termwise matrices, so extract them first:

```
cd data
tar -xzf raw_runs_tfim.tar.gz
tar -xzf raw_runs_lfim.tar.gz
cd ..
```

Then run the analysis scripts from the `analysis/` directory (all paths in
the scripts are relative to it):

```
cd analysis
python step1_q_scaling_tfim.py
python step2_b2_scaling_tfim.py
python step3_decomposition_tfim.py
python step4_coupling_tfim.py
python step5_sign_alignment_tfim.py
```

and likewise for the `_lfim.py` versions. All bootstrap and permutation
procedures use fixed random seeds, so the outputs written to
`results/tfim/` and `results/lfim/` reproduce the CSV files included in
the repository (up to negligible floating-point differences across
platforms). Figure windows may open during execution; close them to
continue.

### Analysis outputs

| Step | Output files (TFIM names; LFIM versions carry the `lfim_` prefix) |
| --- | --- |
| 1 | `q_scaling.csv`, `q_scaling_fits.csv` |
| 2 | `b2_scaling.csv`, `b2_scaling_fits.csv` |
| 3 | `bridge_decomposition_summary.csv`, `bridge_decomposition_steps.csv`, `m2_f_scaling_fits.csv`, `mgr_bias_corrected.csv` |
| 4 | `b2_q_coupling_decomposition.csv` |
| 5 | `sign_correlation_summary_tol{1e-12,1e-10,1e-09,1e-08}.csv`, `sign_zero_fractions_tol{...}.csv`, `threshold_stability_comparison.csv`, `hva_theoretical_zero_magnitudes.csv` |

### Note on the Step 5 zero threshold

Step 5 classifies termwise gradient entries as zero below a magnitude
threshold. The final analyses use `ZERO_TOL = 1e-10`, selected from a
sensitivity sweep rather than by convention: the magnitudes at
theoretically-zero positions (parameters whose generators commute with a
Hamiltonian sector) cluster several orders of magnitude below all remaining
entries, the resulting structural-zero fractions match the exact theoretical
values (1/(2d) for TFIM, 2/(3d) for LFIM), and all reported statistics are
unchanged across thresholds 1e-10 to 1e-8. The sweep outputs documenting
this are included in `results/`.

## Data Formats

- `results/{tfim,lfim}/per_run.csv`: one row per
  (n_qubits, depth, variant, seed, parameter) with the termwise
  diagnostics (`R_k`, `N_eff_k`, `B_eff_k`, `grad_k`, `abs_sum_k`, ...).
- `data/raw_runs_*/n{n}_d{d}_{variant}_seed{ss}_grad_terms.npz`: key
  `grad_terms`, array of shape (n_params, n_terms) holding the termwise
  gradient contributions a_{alpha,k} for one run.

## Versions

- `v1`: original submission version (arXiv v1). Contains the experiment
  drivers only.
- `main` (to be tagged `v2`): adds the full analysis pipeline, result CSVs,
  and raw termwise data accompanying the revised manuscript.

## Citation

If you use this code, please cite the accompanying paper:

```
@misc{kang:2026:bp-di,
      title={Barren Plateaus as Destructive Interference: A Diagnostic
             Framework and Implications for Structured Ansatzes},
      author={Pilsung Kang},
      year={2026},
      eprint={2605.01319},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2605.01319},
}
```

## License

This code is released under the MIT License. See the LICENSE file for
details.
