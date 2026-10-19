#!/bin/bash
# Antioch sync skips git-ignored files even when they are committed. Copy the committed-but-ignored
# training inputs under hope_training/ (e.g. assets/venue_tuple/*.npz question banks) to the session.
# Run from antioch_sim/ after `antioch session new`; sync never touches copied files afterwards.
# With several sessions open: SESSION=ses-... ./push_ignored.sh
set -euo pipefail
cd "$(dirname "$0")/.."
files=$(git ls-files hope_training | while read -r f; do git check-ignore -q --no-index "$f" && echo "$f"; done || true)
cd antioch_sim  # antioch commands need the project folder
for f in $files; do .venv/bin/antioch service cp ${SESSION:+--session "$SESSION"} "../$f" "sim:/workspace/project/$f" | tail -1; done
