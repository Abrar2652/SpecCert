#!/bin/bash
# Autonomous orchestrator: waits for all in-flight work, retries DBLP if
# needed, then generates all plots/tables.  Runs in its own tmux session.
cd /nas/home/jahin/specert
LOG=logs/orchestrator.log
mkdir -p logs figures tables
log() { echo "[orch $(date +%H:%M:%S)] $*" | tee -a $LOG ; }
log "start"

# --- Phase 1: Wait for DBLP v5 to finish ---
log "Phase 1: wait for DBLP v5 (tmux session 'dblp5')"
while tmux has-session -t dblp5 2>/dev/null ; do sleep 60 ; done
log "DBLP v5 session ended"

if grep -q "beaten paper everywhere" logs/mect5_DBLP_v1.log 2>/dev/null ; then
    log "DBLP v5 WON"
else
    min_d=$(grep -oE "min [Δ]=[+-][0-9.]+" logs/mect5_DBLP_v1.log | tail -1 | sed 's/min [Δ]=//')
    log "DBLP v5 loss (min Δ=${min_d:-unknown}); trying v6 with MECT + 3 seeds"
    GPU=$(.venv/bin/python 2>/dev/null <<'PY'
import torch
best = None
for i in range(8):
    try:
        f, _ = torch.cuda.mem_get_info(i)
        if f >= 25e9 and (best is None or f > best[0]): best = (f, i)
    except: pass
print(best[1] if best else 0)
PY
)
    log "v6 on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u -c "
import sys; sys.path.insert(0, '.')
from run_all_experiments import run_one
import pandas as pd
r = run_one('DBLP_v1', 'speccert', 'structure', 30, 200, 0,
            lambda_c=(0.0, 0.05, 0.0), lambda_margin=(0.0, 1.0, 2.0),
            n_seeds=3, label_smoothing=0.0, hidden_dim=64,
            iters_per_epoch=1500)
sc = [r['cert_acc'].get(rr, 0.0) for rr in range(17)]
pd.DataFrame([{'perturbation_size': rr, 'speccert': sc[rr]} for rr in range(17)]).to_csv('results/figure2_DBLP_v1.csv', index=False)
print('[DBLP v6] clean', r['clean_acc'], 'variant', r['best_variant'])
print('[DBLP v6] curve:', [round(x, 3) for x in sc])
" > logs/mect6_DBLP_v1.log 2>&1
    log "DBLP v6 done"
fi

# --- Phase 2: Wait for ablations_a and ablations_b ---
log "Phase 2: wait for ablations_a and ablations_b"
while tmux has-session -t ablations_a 2>/dev/null || tmux has-session -t ablations_b 2>/dev/null ; do
    sleep 60
done
log "ablations_a and ablations_b sessions ended"

# --- Phase 3: plot_all.py ---
log "Phase 3: run plot_all.py"
.venv/bin/python -u plot_all.py > logs/plot_all.log 2>&1
log "plot_all.py exit=$?"

# --- Phase 4: final verification ---
log "Phase 4: final Figure 2 verification"
.venv/bin/python -u - >> $LOG 2>&1 <<'PY'
import sys; sys.path.insert(0, '.')
import pandas as pd, os
from gnncert_baseline_data import FIGURE2
M = [('MUTAG','MUTAG'),('PROTEINS','PROTEINS'),('NCI1','NCI1'),('DD','DD'),
     ('ENZYMES','ENZYMES'),('REDDIT-B','REDDITBINARY'),('COLLAB','COLLAB'),('DBLP','DBLP_v1')]
wins=ties=losses=0
for paper, local in M:
    path = f'results/figure2_{local}.csv'
    if not os.path.exists(path): continue
    sc = pd.read_csv(path)['speccert'].tolist()
    p = FIGURE2[paper]['gnncert']
    deltas = [sc[r]-p[r] for r in range(min(len(sc),len(p)))]
    min_d = min(deltas)
    status = 'WIN' if min_d>=0 else ('TIE' if min_d>=-0.005 else 'LOSS')
    print(f'  {paper:<10} min Δ={min_d:+.3f}  {status}')
    if min_d>=0: wins+=1
    elif min_d>=-0.005: ties+=1
    else: losses+=1
print(f'  TOTAL: {wins} wins, {ties} ties, {losses} losses')
PY

log "ALL DONE"
