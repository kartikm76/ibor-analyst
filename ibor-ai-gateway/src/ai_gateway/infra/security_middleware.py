"""Pure-ASGI security middleware.

Uses raw ASGI interface instead of BaseHTTPMiddleware.
BaseHTTPMiddleware is incompatible with StreamingResponse (SSE) when
the body generator takes >1s to produce its first chunk — it tears
down the body channel before chunks arrive, returning 0 bytes.
Pure ASGI passes each chunk through immediately without buffering.
"""

from __future__ import annotations

import json
import time
import logging

from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from ai_gateway.infra.security import (
    rate_limiter,
    quota_tracker,
    request_logger,
    input_validator,
)
from ai_gateway.config.settings import settings

log = logging.getLogger(__name__)


class SecurityMiddleware:
    """Pure ASGI: API key check, rate limiting, request logging, rate-limit headers."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        path = scope["path"]
        method = scope.get("method", "")

        if path in ["/health", "/docs", "/openapi.json", "/"]:
            await self.app(scope, receive, send)
            return

        # 1. API KEY CHECK
        if settings.api_key:
            if request.headers.get("X-API-Key", "") != settings.api_key:
                await JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key."},
                )(scope, receive, send)
                return

        # 2. CLIENT IP
        client_ip = request.client.host if request.client else "unknown"
        if x_fwd := request.headers.get("X-Forwarded-For"):
            client_ip = x_fwd.split(",")[0].strip()

        # 3. RATE LIMITING
        if settings.rate_limit_enabled and not await rate_limiter.is_allowed(client_ip):
            rs = await rate_limiter.get_status(client_ip)
            await JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded: {settings.rate_limit_requests_per_minute} req/min",
                    "remaining": rs["remaining"],
                    "requests_this_minute": rs["requests_this_minute"],
                },
            )(scope, receive, send)
            return

        # 4. INTERCEPT SEND — adds rate-limit response headers, captures status
        start_time = time.time()
        response_status = [200]

        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start":
                response_status[0] = message.get("status", 200)
                headers = MutableHeaders(scope=message)
                headers.append("X-RateLimit-Limit", str(settings.rate_limit_requests_per_minute))
                if settings.rate_limit_enabled:
                    rs = await rate_limiter.get_status(client_ip)
                    headers.append("X-RateLimit-Remaining", str(rs["remaining"]))
            await send(message)

        await self.app(scope, receive, send_wrapper)

        # 5. POST-REQUEST LOG — runs only after stream is fully sent
        duration_ms = (time.time() - start_time) * 1000
        await request_logger.log_request(
            client_ip=client_ip,
            endpoint=path,
            method=method,
            response_status=response_status[0],
            response_time_ms=duration_ms,
        )


class InputValidationMiddleware:
    """Pure ASGI: validate chat input (question length, banned words, portfolio code)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if scope.get("method") != "POST" or "/chat" not in scope["path"]:
            await self.app(scope, receive, send)
            return

        # Read full body (may arrive in multiple ASGI chunks)
        body_bytes = b""
        more_body = True
        while more_body:
            message = await receive()
            body_bytes += message.get("body", b"")
            more_body = message.get("more_body", False)

        # Replay body to inner app — ASGI body can only be consumed once.
        # After replay, forward subsequent calls to the original receive so that
        # Starlette's listen_for_disconnect task blocks properly until the client
        # actually disconnects (instead of seeing a fake instant disconnect).
        body_replayed = [False]

        async def replay_receive() -> dict:
            if not body_replayed[0]:
                body_replayed[0] = True
                return {"type": "http.request", "body": body_bytes, "more_body": False}
            return await receive()

        async def reject(status: int, detail: str) -> None:
            await JSONResponse(status_code=status, content={"detail": detail})(
                scope, replay_receive, send
            )

        if not body_bytes:
            await reject(400, "Request body is empty.")
            return

        try:
            body_dict = json.loads(body_bytes)
        except json.JSONDecodeError:
            await reject(400, "Invalid JSON in request body.")
            return
        except Exception as exc:
            log.warning("InputValidationMiddleware parse error: %s", exc)
            await self.app(scope, replay_receive, send)
            return

        question = body_dict.get("question", "").strip()
        is_valid, error_msg = input_validator.validate_question(question)
        if not is_valid:
            await reject(400, error_msg)
            return

        portfolio_code = body_dict.get("portfolio_code", "P-ALPHA").strip()
        is_valid, error_msg = input_validator.validate_portfolio_code(portfolio_code)
        if not is_valid:
            await reject(400, f"Invalid portfolio code: {error_msg}")
            return

        await self.app(scope, replay_receive, send)


class QuotaCheckMiddleware:
    """Pure ASGI: enforce in-memory daily question quotas."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if scope.get("method") != "POST" or "/chat" not in scope["path"]:
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        client_ip = request.client.host if request.client else "unknown"
        if x_fwd := request.headers.get("X-Forwarded-For"):
            client_ip = x_fwd.split(",")[0].strip()

        if not await quota_tracker.check_question_quota(client_ip):
            usage = await quota_tracker.get_daily_usage(client_ip)
            await JSONResponse(
                status_code=429,
                content={
                    "detail": f"Daily question limit exceeded ({settings.max_questions_per_day}/day)",
                    "today_usage": usage,
                },
            )(scope, receive, send)
            return

        # Capture response status; record quota usage after stream completes
        status_code = [None]

        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start":
                status_code[0] = message.get("status", 200)
            await send(message)

        await self.app(scope, receive, send_wrapper)

        if status_code[0] in [200, 201]:
            await quota_tracker.record_question(client_ip, 1000)
