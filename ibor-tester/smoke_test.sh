#!/usr/bin/env bash
# Run the smoke test suite against the live stack.
# Usage: ./ibor-tester/smoke_test.sh
# Run this after every Docker rebuild or code change.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
GATEWAY_DIR="$REPO_ROOT/ibor-ai-gateway"

echo "=== IBOR Smoke Tests ==="
echo ""

# Wait for gateway to be healthy before running tests
echo "Waiting for gateway..."
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do
  sleep 2
done
echo "Gateway ready."
echo ""

cd "$GATEWAY_DIR"
uv run pytest tests/smoke/ -v --tb=short "$@"
