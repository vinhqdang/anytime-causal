#!/usr/bin/env bash
# Wait for the main eval run to finish, then run the expanded ablations and
# regenerate all manuscript tables + figures from the full-run JSON.
set -e
cd /c/work/anytime-causal
export PYTHONPATH=.
PY="C:/Users/vinh.dq4/AppData/Local/anaconda3/envs/py313/python.exe"

echo "[finalize] waiting for main run TOTAL..."
until grep -q "TOTAL suite" results/full_run.log 2>/dev/null; do sleep 15; done
echo "[finalize] main run done; launching expanded ablations"

"$PY" -u -m experiments.exp_ablations > results/ablations_full.log 2>&1
echo "[finalize] ablations done"

"$PY" -m experiments.make_tables > results/make_tables.log 2>&1
"$PY" -m experiments.make_figures > results/make_figures.log 2>&1
echo "[finalize] FINALIZE_DONE"
