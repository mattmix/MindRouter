#!/usr/bin/env bash
############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_harness/run_in_container.sh: Run the DLP harness CLI
# inside the running MindRouter app container.
#
# This is the primary way to run the harness on the
# PRODUCTION host (/opt/mindrouter): the app container
# already has every harness dependency (httpx, pymysql,
# gliner with a warm model cache) and, via host networking,
# reaches the gateway at 127.0.0.1:8000 and MariaDB at
# 127.0.0.1:3306 with the container's own DATABASE_URL.
#
# Usage (from the repo root, e.g. /opt/mindrouter):
#   ./dlp_harness/run_in_container.sh corpus --profile accuracy --size 500 --seed 42
#   ./dlp_harness/run_in_container.sh db-check
#   ./dlp_harness/run_in_container.sh e2e --corpus ... --base-url http://127.0.0.1:8000 \
#       --api-key ... --admin-key ... --allow-prod
#
# Run artifacts are copied back to ./dlp_harness_runs/ on
# the host after every invocation.
#
############################################################
set -euo pipefail
cd "$(dirname "$0")/.."

# rm first: `docker compose cp DIR app:EXISTING_DIR` nests instead of
# replacing, silently leaving the PREVIOUS run's harness code on
# PYTHONPATH (mirrors offline_eval.py; rm -rf on a missing path is a
# no-op, so this is safe under set -e)
docker compose exec -T app rm -rf /tmp/dlp_harness
docker compose cp dlp_harness app:/tmp/dlp_harness >/dev/null
docker compose exec -T app python -m compileall -q /tmp/dlp_harness || true

set +e
docker compose exec -T -e PYTHONPATH=/tmp app python -m dlp_harness "$@"
rc=$?
set -e

mkdir -p dlp_harness_runs
docker compose cp app:/tmp/dlp_harness_runs/. dlp_harness_runs/ >/dev/null 2>&1 || true
exit $rc
