#!/usr/bin/env bash
# Final experiments: 10 seeds x 4 conditions (3 EAs + random search).
#
# Run from the repository root, after calibrate_k.py has chosen the control's k:
#     K_CONTROL=7 bash assignments/assignment_1/run_all.sh
# Optional: SEEDS="0 1 2" to run a subset; OVERWRITE=1 to replace existing runs
# (without it, the scripts stop instead of overwriting results).
set -euo pipefail

K_CONTROL="${K_CONTROL:?set K_CONTROL to the tournament size chosen by calibrate_k.py}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
OVERWRITE_FLAG="${OVERWRITE:+--overwrite}"
DIR="assignments/assignment_1"

for seed in $SEEDS; do
    uv run python "$DIR/ea_tree.py" --selection tournament --k 2 --seed "$seed" $OVERWRITE_FLAG
    uv run python "$DIR/ea_tree.py" --selection lexicase --seed "$seed" $OVERWRITE_FLAG
    uv run python "$DIR/ea_tree.py" --selection tournament --k "$K_CONTROL" --seed "$seed" $OVERWRITE_FLAG
    uv run python "$DIR/random_search.py" --seed "$seed" $OVERWRITE_FLAG
done
