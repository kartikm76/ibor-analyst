from __future__ import annotations

import logging
from typing import Any, Dict

import httpx
from fastapi import APIRouter, Request

from ai_gateway.config.settings import settings

log = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    overall = "ok"

    # ── Anthropic API key ────────────────────────────────────────────────────
    key = settings.anthropic_api_key
    if not key or "placeholder" in key or key.endswith("..."):
        checks["anthropic"] = {
            "status": "misconfigured",
            "detail": "ANTHROPIC_API_KEY is missing or still a placeholder",
        }
        overall = "degraded"
    else:
        checks["anthropic"] = {"status": "ok", "model": settings.anthropic_model}

    # ── Spring Boot middleware ────────────────────────────────────────────────
    _base = settings.structured_api_base
    middleware_base = _base[: _base.rfind("/api")] if "/api" in _base else _base.rstrip("/")
    try:
        async with httpx.AsyncClient(verify=False, timeout=3.0) as client:
            r = await client.get(f"{middleware_base}/actuator/health")
            if r.status_code == 200:
                body = r.json()
                checks["middleware"] = {"status": "ok", "detail": body.get("status", "UP")}
            else:
                checks["middleware"] = {"status": "down", "http_status": r.status_code}
                overall = "degraded"
    except Exception as exc:
        checks["middleware"] = {"status": "unreachable", "detail": str(exc)}
        overall = "degraded"
        log.warning("middleware health check failed: %s", exc)

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    pg_pool = getattr(request.app.state, "pg_pool", None)
    if pg_pool is None:
        checks["postgres"] = {"status": "unknown", "detail": "pg_pool not attached to app.state"}
    else:
        try:
            with pg_pool.connection() as conn:
                conn.execute("SELECT 1")
            checks["postgres"] = {"status": "ok"}
        except Exception as exc:
            checks["postgres"] = {"status": "down", "detail": str(exc)}
            overall = "degraded"
            log.warning("postgres health check failed: %s", exc)

    return {"status": overall, "checks": checks}
