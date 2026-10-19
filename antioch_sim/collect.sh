#!/bin/bash
# Copy checkpoints home while jobs run, then release each session when its run is over.
#   ./collect.sh SESSION:RUN_NAME[:TOTAL_ITERS] ...   (default TOTAL_ITERS 44300; every 15 min)
# Copies model_N.pt for N % EVERY == 0 (500) and the final model_<TOTAL-1>.pt, plus the run log, to
# ../artifacts/antioch/<RUN_NAME>/. A session is released once train.py has exited and the files
# are home. Session files are deleted on release, so nothing is released before it is copied.
cd "$(dirname "$0")"
A=.venv/bin/antioch; EVERY=${EVERY:-500}; WB=/workspace/project/hope_training/whole_body_tracking
while :; do
  left=0
  for job in "$@"; do
    IFS=: read -r ses run total <<< "$job"; total=${total:-44300}
    out=../artifacts/antioch/$run; mkdir -p "$out"
    [ -f "$out/.collected" ] && continue
    left=1
    info=$($A service exec --session "$ses" --no-stream -- bash -c "
      pgrep -f scripts/train.py >/dev/null && echo RUNNING || echo STOPPED
      ls $WB/logs/rsl_rl/*/*_$run/model_*.pt 2>/dev/null" 2>/dev/null | grep -v '^Session ses-')
    if [ -z "$info" ]; then
      echo x >> "$out/.unreachable"; echo "[collect] $(date +%H:%M) $run: session not reachable"
      (( $(wc -l < "$out/.unreachable") >= 4 )) && { echo "[collect] $run: gave up (session gone?)"; touch "$out/.collected"; }
      continue
    fi
    rm -f "$out/.unreachable"
    state=$(echo "$info" | head -1)
    for f in $(echo "$info" | grep '\.pt$'); do
      n=$(basename "$f" .pt); n=${n#model_}
      if (( n % EVERY == 0 || n == total - 1 )) && [ ! -s "$out/model_$n.pt" ]; then
        $A service cp --session "$ses" "sim:$f" "$out/model_$n.pt" >/dev/null 2>&1 && echo "[collect] $(date +%H:%M) $run: model_$n.pt"
      fi
    done
    $A service cp --session "$ses" "sim:$WB/logs/antioch/$run.log" "$out/$run.log" >/dev/null 2>&1
    if [ "$state" = STOPPED ] && [ -s "$out/model_$((total - 1)).pt" ]; then
      $A session release "$ses" >/dev/null 2>&1 && echo "[collect] $(date +%H:%M) $run: finished, files home, session released"
      touch "$out/.collected"
    elif [ "$state" = STOPPED ]; then
      echo "[collect] $(date +%H:%M) $run: train.py stopped before the final checkpoint -> NOT releasing; check $out/$run.log"
      touch "$out/.collected"
    fi
  done
  [ $left = 0 ] && break
  sleep 900
done
echo "[collect] all jobs handled"
