#!/bin/bash
# Start one training job on its own Antioch session. The job settings go in as session env vars;
# the service command (antioch.yaml) waits for /workspace/project/.train_go, which this script creates
# after uploading the ignored inputs and the warm-start checkpoint. Prints the session id.
#   ./start_job.sh TASK RUN_NAME TOTAL_ITERS [LOCAL_INIT_CHECKPOINT] [NUM_ENVS] [IDLE_TIMEOUT]
set -euo pipefail
cd "$(dirname "$0")"
TASK=$1; RUN=$2; TOTAL=$3; INIT=${4:-}; ENVS=${5:-4096}; IDLE=${6:-1d}
A=.venv/bin/antioch
remote_init=""
if [ -n "$INIT" ]; then remote_init=/workspace/project/hope_training/whole_body_tracking/checkpoints_in/$(basename "$INIT"); fi
SES=$($A session new --idle-timeout "$IDLE" --env TASK="$TASK" --env RUN_NAME="$RUN" --env TOTAL_ITERS="$TOTAL" \
  --env NUM_ENVS="$ENVS" --env INIT_CHECKPOINT="$remote_init" --json | python3 -c "import json,sys; print(json.load(sys.stdin)['session_id'])")
echo "[start_job] $RUN on $SES"
SESSION=$SES ./push_ignored.sh
if [ -n "$INIT" ]; then $A service cp --session "$SES" "$INIT" "sim:$remote_init" | tail -1; fi
# exec syncs the source first; then release the waiting service command
$A service exec --session "$SES" --no-stream -- touch /workspace/project/.train_go
echo "$SES"
