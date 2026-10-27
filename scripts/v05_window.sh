#!/usr/bin/env bash
# v0.5 stage driver for one 8-GPU Linux host. Each model process sees exactly one GPU.
# Usage: bash scripts/v05_window.sh <stage>   (stages below; every stage is resumable)
set -uo pipefail
cd "$(dirname "$0")/.."
: "${STOP:?set STOP, e.g. 2026-10-12T00:00:00+08:00}"
: "${RUN:?set RUN, e.g. outputs/v05/h20-20261011}"
PY=${PY:-.venv/bin/python}
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
LOGS=outputs/logs/v05
mkdir -p "$LOGS"
BOUND=(bash docs/h20_deadline.sh --stop-at "$STOP")

# Run "name|args" jobs over the GPU list, at most one process per GPU; record exit codes.
queue() {
  local -a jobs=("$@") free=("${GPUS[@]}") pids=() names=() devices=()
  local i
  while ((${#jobs[@]})) || ((${#pids[@]})); do
    while ((${#jobs[@]})) && ((${#free[@]})); do
      local name=${jobs[0]%%|*} args=${jobs[0]#*|} device=${free[0]}
      jobs=("${jobs[@]:1}") free=("${free[@]:1}")
      # shellcheck disable=SC2086
      "${BOUND[@]}" env CUDA_VISIBLE_DEVICES="$device" "$PY" -m exec_feedback.rft $args --run "$RUN" --execute \
        > "$LOGS/$name.log" 2>&1 &
      pids+=($!) names+=("$name") devices+=("$device")
      echo "$(date '+%F %T') start $name on GPU $device"
    done
    wait -n 2>/dev/null || true
    for i in "${!pids[@]}"; do
      if ! kill -0 "${pids[$i]}" 2>/dev/null; then
        wait "${pids[$i]}"; local rc=$?
        echo "$(date '+%F %T') done $((rc)) ${names[$i]}" | tee -a "$LOGS/exit-codes.txt"
        free+=("${devices[$i]}")
        unset 'pids[i]' 'names[i]' 'devices[i]'
      fi
    done
    pids=("${pids[@]}") names=("${names[@]}") devices=("${devices[@]}")
  done
}

stage=${1:?stage: sample1|train1|eval1|prefer|evalP|sample2|train2|eval2dev|eval2|report}
case "$stage" in
  sample1|sample2)
    round=${stage#sample}; policy=${POLICY:-B}; jobs=()
    for s in "${!GPUS[@]}"; do jobs+=("sample-r$round-s$s|sample --round $round --policy $policy --shard $s --shards ${#GPUS[@]}"); done
    queue "${jobs[@]}" ;;
  train1)
    queue "train-R-r1-s42|train --round 1 --variant R --seed 42" "train-R-r1-s43|train --round 1 --variant R --seed 43" \
          "train-R-r1-s44|train --round 1 --variant R --seed 44" "train-Rclean-r1-s42|train --round 1 --variant Rclean --seed 42" \
          "evalB-dev|evaluate --condition B --role dev" "evalB-validation|evaluate --condition B --role validation" \
          "evalB-test|evaluate --condition B --role test" ;;
  eval1)
    # P training needs only the frozen R adapters, so it shares the GPUs with R's evaluations.
    jobs=("prefer-s42|prefer --seed 42" "prefer-s43|prefer --seed 43" "prefer-s44|prefer --seed 44")
    for role in dev validation test; do
      for c in R_r1_seed42 R_r1_seed43 R_r1_seed44 Rclean_r1_seed42; do jobs+=("eval-$c-$role|evaluate --condition $c --role $role"); done
    done
    queue "${jobs[@]}" ;;
  prefer)
    queue "prefer-s42|prefer --seed 42" "prefer-s43|prefer --seed 43" "prefer-s44|prefer --seed 44" ;;
  evalP)
    jobs=()
    for role in dev validation test; do
      for c in P_r1_seed42 P_r1_seed43 P_r1_seed44; do jobs+=("eval-$c-$role|evaluate --condition $c --role $role"); done
    done
    queue "${jobs[@]}" ;;
  train2)
    queue "train-R-r2-s42|train --round 2 --variant R --seed 42" "train-R-r2-s43|train --round 2 --variant R --seed 43" \
          "train-R-r2-s44|train --round 2 --variant R --seed 44" ;;
  eval2dev)
    queue "eval-R_r2_seed42-dev|evaluate --condition R_r2_seed42 --role dev" \
          "eval-R_r2_seed43-dev|evaluate --condition R_r2_seed43 --role dev" \
          "eval-R_r2_seed44-dev|evaluate --condition R_r2_seed44 --role dev" ;;
  eval2)
    jobs=()
    for role in validation test; do
      for c in R_r2_seed42 R_r2_seed43 R_r2_seed44; do jobs+=("eval-$c-$role|evaluate --condition $c --role $role"); done
    done
    queue "${jobs[@]}" ;;
  report)
    "$PY" -m exec_feedback.rft report --run "$RUN" --out "$RUN/report" --execute > "$LOGS/report.log" 2>&1
    echo "report exit $?" ;;
  *) echo "unknown stage $stage" >&2; exit 2 ;;
esac
