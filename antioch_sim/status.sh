#!/bin/bash
# Progress of training jobs on Antioch sessions: ./status.sh SESSION:RUN_NAME [SESSION:RUN_NAME ...]
cd "$(dirname "$0")"
for job in "$@"; do
  ses=${job%%:*}; run=${job#*:}
  echo "== $run ($ses)"
  .venv/bin/antioch service exec --session "$ses" --no-stream -- bash -c "
    L=/workspace/project/hope_training/whole_body_tracking/logs/antioch/$run.log
    pgrep -f "[s]cripts/train.py" >/dev/null && echo running || echo 'NOT running'
    grep -E 'Learning iteration' \$L | tail -1 | tr -s ' '
    grep -E 'Iteration time|Mean value_function loss|Mean episode length' \$L | tail -3 | tr -s ' '
    grep -E 'Traceback|ERROR during run|watchdog' \$L | head -3
    for k in strike_vel_pass_exact virtual_hit_rate virtual_land_inbounds_rate foot_displacement_max foot_planar_speed_max root_height com_support_offset; do
      grep -E \"Metrics/racket_target/\$k:\" \$L | tail -200 | awk -v k=\$k '{s+=\$NF;n++} END{if(n) printf \"  %s (last %d logs) %.3f\\n\", k, n, s/n}'
    done
    grep -E 'Episode_Termination/(feet_planted|base_fell_tilt|base_too_low|time_out):' \$L | tail -4 | tr -s ' '
  " 2>&1 | grep -v '^Session ses-'
done
