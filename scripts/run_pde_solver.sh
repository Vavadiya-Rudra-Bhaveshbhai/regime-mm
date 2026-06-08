#!/usr/bin/env bash
# Solve HJB PDEs and save spread tables to results/
set -e
cd "$(dirname "$0")/.."
pip install -r requirements.txt -q
python pde_solver/src/hjb_solver.py --config configs/default.yaml --out results
python pde_solver/src/visualize_pde.py --results results --plots results/plots
echo "Done. Plots in results/plots/"
