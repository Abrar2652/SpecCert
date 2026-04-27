#!/bin/bash
# Queue A: figure4 (5 missing datasets) then figure7 (4 datasets, trimmed).
cd /nas/home/jahin/specert
GPU=${1:-4}
MISSING4="NCI1 DD ENZYMES REDDITBINARY COLLAB"
FIG7="MUTAG PROTEINS NCI1 COLLAB"

echo "[A $(date +%T)] figure4 on: $MISSING4"
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u run_all_experiments.py \
    --experiment figure4 --device 0 --epochs 150 --datasets $MISSING4 \
    > logs/ablation_figure4.log 2>&1
echo "[A $(date +%T)] figure4 done"

echo "[A $(date +%T)] figure7 on: $FIG7 (trimmed)"
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u run_all_experiments.py \
    --experiment figure7 --device 0 --epochs 150 --datasets $FIG7 \
    > logs/ablation_figure7.log 2>&1
echo "[A $(date +%T)] figure7 done"
echo "[A $(date +%T)] QUEUE A COMPLETE"
