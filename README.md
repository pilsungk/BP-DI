# BP-DI

This repository contains experiment scripts for comparing the hardware-efficient ansatz (HEA) and the Hamiltonian variational ansatz (HVA) using destructive-interference diagnostics.

At the current stage, the repository includes two experiment drivers:

- `bp_hva_vs_hea_save_per_run_v3.py`
- `bp_hva_vs_hea_lfim_save_per_run_v1.py`

## Overview

This codebase supports the experimental comparison of HEA and HVA under two closely related Ising-model settings:

1. **TFIM**
   - Script: `bp_hva_vs_hea_save_per_run_v3.py`
   - Goal: compare HEA and HVA on the transverse-field Ising model using destructive-interference diagnostics

2. **LFIM**
   - Script: `bp_hva_vs_hea_lfim_save_per_run_v1.py`
   - Goal: compare HEA and HVA on a longitudinal-field extension of the Ising model using the same diagnostic framework

## Included Scripts

### `bp_hva_vs_hea_save_per_run_v3.py`

Compares HEA and HVA on the transverse-field Ising model (TFIM).

Main outputs:
- `per_run.csv`
- `grad_mean_check.csv`
- `layer_summary.csv`
- `structure_compare.csv`
- `summary.json`
- `figures/*.pdf`

Ansatz definitions:
- **HEA**: each layer applies trainable `RY-RZ` rotations on every qubit, followed by an entangler ring
- **HVA**: each layer applies
  - `exp(-i beta sum X_i)`
  - then `exp(-i gamma sum Z_i Z_{i+1})`

Implemented exactly using commuting gate products:
- `RX(2*beta)` on each qubit
- `RZZ(2*gamma)` on each edge

Example:
```bash
python bp_hva_vs_hea_save_per_run_v3.py \
    --n_qubits 4,6 \
    --depths 4,6 \
    --seeds 10 \
    --outdir_root ./runs \
    --save_raw_terms false

bp_hva_vs_hea_lfim_save_per_run_v1.py
Compares HEA and HVA on the longitudinal-field Ising model (LFIM).
Hamiltonian:
H = - sum_i Z_i Z_{i+1} + hx sum_i X_i + hz sum_i Z_i   (OBC)

Main outputs:

per_run.csv
layer_summary.csv
structure_compare.csv
summary.json
figures/*.pdf

Ansatz definitions:

HEA: each layer applies trainable RY-RZ rotations on every qubit, followed by an entangler ring
HVA: each layer applies

exp(-i beta_x sum X_i)
then exp(-i gamma_zz sum Z_i Z_i+1)
then exp(-i beta_z sum Z_i)



Implemented exactly using commuting gate products:

RX(2*beta_x) on each qubit
RZZ(2*gamma_zz) on each edge
RZ(2*beta_z) on each qubit

Example:
python bp_hva_vs_hea_lfim_save_per_run_v1.py \
    --n_qubits 4,6 \
    --depths 4,6 \
    --seeds 10 \
    --outdir_root ./runs \
    --save_raw_terms false

Default Run
Each script can also be executed with its default settings:
python bp_hva_vs_hea_save_per_run_v3.py
python bp_hva_vs_hea_lfim_save_per_run_v1.py

Notes

Both scripts are designed to save run-level results for later aggregation and analysis.
The repository is currently minimal and contains only the core experiment scripts.
Please refer to the script headers for the current command-line options and output details.

Status
This repository is currently under active development.
