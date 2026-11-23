#!/usr/bin/env bash
# Starts the Docker stack and exposes it on a temporary public https URL (Cloudflare quick tunnel).
# Usage: scripts/public_url.sh [api_containers]   (default 2). Stop with Ctrl+C.
set -euo pipefail
N=${1:-2}
docker compose up -d --build --scale api="$N"
until curl -sf http://localhost:8080/api/health >/dev/null; do sleep 2; done
echo "App is up locally on http://localhost:8080 with $N API containers. Opening public tunnel..."
echo "Look for the https://<random>.trycloudflare.com line below."
cloudflared tunnel --url http://localhost:8080
