#!/usr/bin/env bash
# Usage: scripts/run_load_tests.sh [users] [spawn_rate] [seconds]  (uses docker compose + Locust)
set -euo pipefail
USERS=${1:-50}; RATE=${2:-10}; DURATION=${3:-60}
mkdir -p results
for N in 1 2 3 4; do
  docker compose down -v >/dev/null 2>&1 || true
  docker compose up -d --build --scale api=$N
  until curl -sf http://localhost:8080/api/health >/dev/null; do sleep 2; done
  curl -s -o /dev/null -F "file=@$(find data/test -name '*.wav' | head -1)" http://localhost:8080/api/predict
  locust -f locustfile.py --headless -u "$USERS" -r "$RATE" -t "${DURATION}s" \
    --host http://localhost:8080 --csv "results/locust_${N}_containers" --only-summary
done
docker compose down -v
python3 scripts/summarize_locust.py
