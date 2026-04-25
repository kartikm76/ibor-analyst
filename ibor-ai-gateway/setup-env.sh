#!/bin/bash
# Frozen environment setup for ibor-ai-gateway
# Recreates Python venv from uv.lock to ensure reproducible dependencies

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info() { echo -e "${YELLOW}==> $*${NC}"; }
ok() { echo -e "${GREEN}✓ $*${NC}"; }
error() { echo -e "${RED}✗ $*${NC}"; exit 1; }

info "Setting up frozen Python environment from uv.lock..."
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    error "uv not found. Install with: brew install uv"
fi

# Remove old venv if exists
if [ -d .venv ]; then
    info "Removing old virtual environment..."
    rm -rf .venv
fi

# Sync from uv.lock (deterministic, frozen dependencies)
info "Creating virtual environment from uv.lock..."
uv sync --frozen

ok "Python environment setup complete"
echo ""
echo "Environment details:"
.venv/bin/python --version
echo ""
echo "Verifying core dependencies..."
.venv/bin/python -c "import anthropic, openai, fastapi, uvicorn, psycopg, pydantic; print('✓ All core dependencies available')" || error "Missing dependencies"
echo ""
ok "Environment is frozen and reproducible"
echo ""

# ── .env check (repo root — single .env for the whole stack) ─────────────────
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
if [ ! -f "$REPO_ROOT/.env" ]; then
    info "No .env found — creating from .env.example..."
    cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
    echo ""
    echo -e "${RED}ACTION REQUIRED: Edit .env and set your real ANTHROPIC_API_KEY${NC}"
    echo "  File: $REPO_ROOT/.env"
    echo ""
else
    if grep -q "sk-ant-\.\.\.your-key-here\.\.\." "$REPO_ROOT/.env" 2>/dev/null; then
        echo -e "${RED}WARNING: .env still has the placeholder ANTHROPIC_API_KEY${NC}"
        echo "  Edit $REPO_ROOT/.env and replace sk-ant-...your-key-here... with your real key."
        echo ""
    else
        ok ".env looks configured"
        echo ""
    fi
fi

echo "Next steps:"
echo "  1. Verify $REPO_ROOT/.env has your real ANTHROPIC_API_KEY"
echo "  2. Run: uv run uvicorn ai_gateway.main:app --host 127.0.0.1 --port 8000 --reload"
echo ""
