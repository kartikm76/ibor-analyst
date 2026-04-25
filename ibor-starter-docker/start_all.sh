#!/bin/bash
# Start all Docker containers (ibor-analyst full stack)
# Usage: ./start_all.sh
#
# This script:
# 1. Builds Docker images
# 2. Starts PostgreSQL with pgvector
# 3. Runs bootstrap container (loads schema + CSV data)
# 4. Starts Spring Boot middleware
# 5. Starts FastAPI gateway
# 6. Starts React UI

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info() { echo -e "${YELLOW}==> $*${NC}"; }
ok() { echo -e "${GREEN}✓ $*${NC}"; }
error() { echo -e "${RED}✗ $*${NC}"; exit 1; }

echo ""
info "Starting all Docker containers..."
echo ""

cd "$ROOT"

# Start all services (docker-compose handles the dependency chain)
info "Building images and starting containers..."
docker-compose up -d

echo ""

# Monitor bootstrap completion
info "Waiting for database bootstrap to complete..."
MAX_WAIT=120
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    BOOTSTRAP_STATUS=$(docker inspect -f '{{.State.Status}}' ibor-bootstrap 2>/dev/null || echo "missing")

    case "$BOOTSTRAP_STATUS" in
        exited)
            EXIT_CODE=$(docker inspect -f '{{.State.ExitCode}}' ibor-bootstrap 2>/dev/null || echo "1")
            if [ "$EXIT_CODE" = "0" ]; then
                ok "Database initialized and data loaded"
                break
            else
                error "Bootstrap failed (exit code: $EXIT_CODE). Check: docker logs ibor-bootstrap"
            fi
            ;;
        missing)
            error "Bootstrap container not found"
            ;;
    esac

    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    error "Bootstrap timed out after ${MAX_WAIT}s"
fi

# Wait a bit for other services to be ready
info "Waiting for all services to be ready..."
sleep 3

# Check if middleware is ready
TIMEOUT=60
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    if curl -s http://localhost:8080/actuator/health >/dev/null 2>&1; then
        ok "Spring Boot middleware is running"
        break
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

# Check if gateway is ready
TIMEOUT=60
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        ok "FastAPI gateway is running"
        break
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

echo ""
echo -e "${GREEN}All Docker services are running!${NC}"
echo ""
echo "Available at:"
echo "  Spring Boot:   http://localhost:8080/swagger-ui.html"
echo "  FastAPI:       http://localhost:8000/docs"
echo "  React UI:      http://localhost:5173"
echo ""
echo "View logs:"
echo "  docker-compose logs -f"
echo ""
echo "To stop all services:"
echo "  bash ibor-starter-docker/stop_all.sh"
echo ""
