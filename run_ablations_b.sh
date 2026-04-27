#!/bin/bash
# Queue B: figure5 (5 missing datasets) then figure8 (4 datasets, trimmed).
cd /nas/home/jahin/specert
GPU=${1:-5}
MISSING5="NCI1 DD ENZYMES REDDITBINARY COLLAB"
FIG8="MUTAG PROTEINS NCI1 COLLAB"

echo "[B $(date +%T)] figure5 on: $MISSING5"
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u run_all_experiments.py \
    --experiment figure5 --device 0 --epochs 150 --datasets $MISSING5 \
    > logs/ablation_figure5.log 2>&1
echo "[B $(date +%T)] figure5 done"

echo "[B $(date +%T)] figure8 on: $FIG8 (trimmed)"
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u run_all_experiments.py \
    --experiment figure8 --device 0 --epochs 150 --datasets $FIG8 \
    > logs/ablation_figure8.log 2>&1
echo "[B $(date +%T)] figure8 done"
echo "[B $(date +%T)] QUEUE B COMPLETE"
