#!/bin/bash
# One-screen training status on the cluster: my jobs, the club's quota use, and the latest iteration
# and key metrics of each training run.
#   bash hope_training/slurm/pip_env/status.sh          # running/queued training jobs
#   bash hope_training/slurm/pip_env/status.sh --all    # also jobs that ended in the last 3 days
#   bash hope_training/slurm/pip_env/status.sh 4382     # one job
set -uo pipefail
LOGS="${HOPE_LOGS:-$HOME/hope/slurm_logs}"
CKPTS="${HOPE_REPO:-$HOME/hope/HOPE}/hope_training/whole_body_tracking/logs/rsl_rl/unitree_g1_hitter_pingpong"
# label|log key
METRICS=(
  "mean reward|Mean reward"
  "racket within 10 cm at strike|Metrics/racket_target/exact_strike_pos_success_10cm"
  "pos+vel+normal all OK at strike|Metrics/racket_target/strike_composite_success_exact"
  "virtual ball: contact rate|Metrics/racket_target/virtual_contact_rate_forehand"
  "virtual ball: legal return rate|Metrics/racket_target/virtual_legal_rate_forehand"
  "virtual ball: landing error (m)|Metrics/racket_target/virtual_landing_error_m_forehand"
  "falls (share of episode ends)|Episode_Termination/base_fell_tilt"
  "rally-pace share of swings|Metrics/motion/rally_pace_swing"
  "virtual-ball timing jitter (s)|Metrics/racket_target/virtual_timing_jitter_s"
  "seconds per iteration|Iteration time"
)

echo "== my jobs"
squeue -u "$USER" -o "%.8i %.20j %.9T %.11M %.5C %.6m %R"
echo
echo "== tritondroids usage (cap: 32 CPU, 96G, 8 GPU)"
squeue -h -A tritondroids -t RUNNING -o "%C %m %u" | awk '
  { cpu += $1; m = $2; sub(/G$/, "", m); mem += m; user[$3]++ }
  END { printf "running: %d CPU, %d G;", cpu, mem; for (u in user) printf " %s=%d", u, user[u]; print "" }'

if [ "${1:-}" = "--all" ]; then
  files=$(find "$LOGS" -name 'hope-g1-*.out' -mtime -3 2>/dev/null | sort)
elif [ -n "${1:-}" ]; then
  files=$(ls "$LOGS"/*-"$1".out 2>/dev/null)
else
  files=""
  for j in $(squeue -h -u "$USER" -o %i); do files="$files $(ls "$LOGS"/*-"$j".out 2>/dev/null)"; done
fi
for f in $files; do
  grep -q "Learning iteration" "$f" || continue
  job=$(basename "$f" .out); job=${job##*-}
  state=$(squeue -h -j "$job" -o %T 2>/dev/null); state=${state:-ENDED}
  header=$(grep -m1 -E "^\[(fh-)?train\] job=" "$f")
  run=$(sed -nE 's/.* run=([^ ]+).*/\1/p' <<<"$header")
  task=$(sed -nE 's/.* task=([^ ]+).*/\1/p' <<<"$header")
  iter=$(grep -oE "Learning iteration [0-9]+/[0-9]+" "$f" | tail -1 | awk '{print $3}')
  echo
  echo "== job $job [$state]  run=${run:-?}  task=${task:-HOPEPingPongUnitreeG1_FH}  iteration ${iter:-?}"
  for entry in "${METRICS[@]}"; do
    label=${entry%%|*}; key=${entry#*|}
    v=$(grep -F "$key:" "$f" | tail -1 | awk -F: '{gsub(/ /, "", $NF); print $NF}')
    [ -n "$v" ] && printf "   %-34s %s\n" "$label" "$v"
  done
  if [ "$state" != "ENDED" ] && [ -n "$run" ]; then
    ckpt=$(ls -1t "$CKPTS"/*_"$run"/model_*.pt 2>/dev/null | head -1)
    [ -n "$ckpt" ] && echo "   newest checkpoint: ${ckpt#"$CKPTS"/}"
  fi
  grep -qE "Traceback|ERROR during run" "$f" && echo "   !! errors in the log: grep -n Traceback $f"
done
exit 0
