#!/usr/bin/env bash
# Operator-invoked Linux wrapper; never rents/releases hardware or extends a lease.
set -euo pipefail
if [[ $# -lt 3 || $1 != --stop-at ]]; then
  echo 'Usage: bash docs/h20_deadline.sh --stop-at ISO_WITH_TIMEZONE command [args...]' >&2
  exit 2
fi
stop_at=$2
shift 2
if [[ ! $stop_at =~ (Z|[+-][0-9]{2}:[0-9]{2})$ ]]; then
  echo 'An explicit timezone is required (Z or +08:00, for example)' >&2
  exit 2
fi
stop_epoch=$(date --date="$stop_at" +%s)
remaining=$((stop_epoch - $(date +%s)))
if (( remaining <= 0 )); then
  echo "Compute deadline already passed: $stop_at; nothing started" >&2
  exit 124
fi
export EFA_STOP_AT_UNIX=$stop_epoch
# Do not use --foreground: timeout must signal the workload's process group.
# INT permits Python cleanup; KILL bounds unresponsive cleanup by 60 seconds.
exec timeout --signal=INT --kill-after=60s "${remaining}s" "$@"
