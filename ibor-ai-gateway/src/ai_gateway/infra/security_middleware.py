"""FastAPI middleware for security enforcement."""

from __future__ import annotations

import time
import logging
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ai_gateway.infra.security import (
    rate_limiter,
    quota_tracker,
    request_logger,
    input_validator,
    cost_tracker,
)
from ai_gateway.config.settings import settings

log = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Enforce rate limiting, request logging, and quota checks.

    Order of operations:
    1. Extract client IP
    2. Check rate limit
    3. Log request start
    4. Call next middleware/endpoint
    5. Log request completion
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client IP (handle X-Forwarded-For for proxies)
        client_ip = request.client.host if request.client else "unknown"
        if x_forwarded := request.headers.get("X-Forwarded-For"):
            client_ip = x_forwarded.split(",")[0].strip()

        # Skip security checks for health/docs endpoints
        if request.url.path in ["/health", "/docs", "/openapi.json", "/"]:
            return await call_next(request)

        # 1. API KEY CHECK
        # Always check the API key — nginx already injects X-API-Key for internal traffic,
        # so the IP-based bypass is unnecessary and was broken (Railway's ingress always
        # appears as 100.64.* regardless of whether the caller is internal or external).
        if settings.api_key:
            request_key = request.headers.get("X-API-Key", "")
            if request_key != settings.api_key:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key."},
                )

        # 2. RATE LIMITING
        if settings.rate_limit_enabled and not await rate_limiter.is_allowed(client_ip):
            rate_status = await rate_limiter.get_status(client_ip)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded: {settings.rate_limit_requests_per_minute} requests per minute",
                    "remaining": rate_status["remaining"],
                    "requests_this_minute": rate_status["requests_this_minute"],
                },
            )

        # 3. REQUEST LOGGING & TIMING
        start_time = time.time()
        question_preview = None

        # Extract question from request for logging
        if request.method == "POST" and "/chat" in request.url.path:
            try:
                body = await request.body()
                if body:
                    import json
                    body_dict = json.loads(body)
                    question_preview = body_dict.get("question", "")
            except Exception:
                pass
            # Reset body stream for endpoint to read
            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}
            request._receive = receive

        # 4. CALL ENDPOINT
        response = await call_next(request)

        # 5. POST-RESPONSE LOGGING
        duration_ms = (time.time() - start_time) * 1000
        await request_logger.log_request(
            client_ip=client_ip,
            endpoint=request.url.path,
            method=request.method,
            question=question_preview,
            response_status=response.status_code,
            response_time_ms=duration_ms,
        )

        # 6. ADD SECURITY HEADERS
        response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_requests_per_minute)
        if settings.rate_limit_enabled:
            rate_status = await rate_limiter.get_status(client_ip)
            response.headers["X-RateLimit-Remaining"] = str(rate_status["remaining"])

        return response


class InputValidationMiddleware(BaseHTTPMiddleware):
    """Validate chat endpoint input before processing."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only validate chat endpoints
        if request.method != "POST" or "/chat" not in request.url.path:
            return await call_next(request)

        try:
            body = await request.body()
            if not body:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Request body is empty."},
                )

            import json
            body_dict = json.loads(body)
            question = body_dict.get("question", "").strip()

            # Validate question
            is_valid, error_msg = input_validator.validate_question(question)
            if not is_valid:
                return JSONResponse(
                    status_code=400,
                    content={"detail": error_msg},
                )

            # Validate portfolio code if provided
            portfolio_code = body_dict.get("portfolio_code", "P-ALPHA").strip()
            is_valid, error_msg = input_validator.validate_portfolio_code(portfolio_code)
            if not is_valid:
                return JSONResponse(
                    status_code=400,
                    content={"detail": f"Invalid portfolio code: {error_msg}"},
                )

            # Reset body stream for endpoint to read
            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}
            request._receive = receive

        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid JSON in request body."},
            )
        except Exception as e:
            log.warning(f"InputValidationMiddleware error: {e}")

        return await call_next(request)


class QuotaCheckMiddleware(BaseHTTPMiddleware):
    """Check quotas before processing expensive requests."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only enforce quotas on chat endpoints
        if request.method != "POST" or "/chat" not in request.url.path:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        if x_forwarded := request.headers.get("X-Forwarded-For"):
            client_ip = x_forwarded.split(",")[0].strip()

        # Check question quota
        if not await quota_tracker.check_question_quota(client_ip):
            usage = await quota_tracker.get_daily_usage(client_ip)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Daily question limit exceeded ({settings.max_questions_per_day} questions per day)",
                    "today_usage": usage,
                },
            )

        # Continue to endpoint
        response = await call_next(request)

        # Record the question after successful completion
        if response.status_code in [200, 201]:
            # Estimate tokens used (rough: 1 token per 4 chars of response)
            estimated_tokens = 1000  # Conservative estimate
            await quota_tracker.record_question(client_ip, estimated_tokens)

        return response
