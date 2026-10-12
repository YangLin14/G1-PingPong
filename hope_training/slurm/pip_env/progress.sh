#!/bin/bash
# Training progress of one job as a markdown table (one row every EVERY iterations, plus the newest),
# ready to paste into docs/TRAINING_STATUS_G1_FOREHAND.md. Requeued jobs log some iterations twice;
# the later entry wins.
#   bash hope_training/slurm/pip_env/progress.sh 4382         # every 500 iterations
#   bash hope_training/slurm/pip_env/progress.sh 4382 250
# Columns: the same metrics as status.sh (racket within 10 cm, all-OK strike, virtual-ball contact,
# legal return, landing error, fall share).
set -uo pipefail
job="${1:?usage: progress.sh JOBID [EVERY]}"; every="${2:-500}"
f=$(ls "${HOPE_LOGS:-$HOME/hope/slurm_logs}"/*-"$job".out 2>/dev/null | head -1)
[ -n "$f" ] || { echo "no log for job $job" >&2; exit 1; }
sed 's/\x1b\[[0-9;]*m//g' "$f" | awk -v every="$every" '
  function val() { v = $NF; gsub(/ /, "", v); return v }
  /Learning iteration [0-9]+\// { split($0, a, "Learning iteration "); split(a[2], b, "/"); it = b[1] + 0; seen[it] = 1; if (it > last) last = it; next }
  /Mean reward:/                                                   { r[it] = val() }
  /Metrics\/racket_target\/exact_strike_pos_success_10cm:/         { p[it] = val() }
  /Metrics\/racket_target\/strike_composite_success_exact:/        { c[it] = val() }
  /Metrics\/racket_target\/virtual_contact_rate_forehand:/         { h[it] = val() }
  /Metrics\/racket_target\/virtual_legal_rate_forehand:/           { l[it] = val() }
  /Metrics\/racket_target\/virtual_landing_error_m_forehand:/      { e[it] = val() }
  /Episode_Termination\/base_fell_tilt:/                           { d[it] = val() }
  END {
    print "| iteration | reward | racket ≤10 cm | all OK | contact | legal return | landing err (m) | falls |"
    print "|---:|---:|---:|---:|---:|---:|---:|---:|"
    for (i = 0; i <= last; i++) if ((i in seen) && (i % every == 0 || i == last))
      printf "| %d | %s | %s | %s | %s | %s | %s | %s |\n", i, r[i], p[i], c[i], h[i], l[i], e[i], d[i]
  }'
