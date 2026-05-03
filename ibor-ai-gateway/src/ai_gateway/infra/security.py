"""Security layer: rate limiting, input validation, quotas, cost tracking."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Optional

from ai_gateway.config.settings import settings

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Rate Limiting (Token Bucket per IP)
# ─────────────────────────────────────────────────────────────────────────────

class RateLimiter:
    """Token bucket rate limiter: allows N requests per minute per IP."""

    def __init__(self, rpm: int = 30):
        self.rpm = rpm
        self.buckets: dict[str, dict[str, Any]] = {}
        self.lock = asyncio.Lock()

    async def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed. Returns False if rate limit exceeded."""
        if not settings.rate_limit_enabled:
            return True

        async with self.lock:
            now = time.time()
            if client_ip not in self.buckets:
                # Initialize bucket: (tokens, last_refill_time)
                self.buckets[client_ip] = {
                    "tokens": self.rpm,
                    "last_refill": now,
                    "requests_this_minute": 0,
                }

            bucket = self.buckets[client_ip]

            # Refill tokens every 60 seconds
            time_elapsed = now - bucket["last_refill"]
            if time_elapsed >= 60:
                bucket["tokens"] = self.rpm
                bucket["last_refill"] = now
                bucket["requests_this_minute"] = 0

            # Check if we have tokens
            if bucket["tokens"] > 0:
                bucket["tokens"] -= 1
                bucket["requests_this_minute"] += 1
                return True

            return False

    async def get_status(self, client_ip: str) -> dict[str, int]:
        """Get remaining tokens and requests this minute."""
        async with self.lock:
            if client_ip not in self.buckets:
                return {"remaining": self.rpm, "requests_this_minute": 0}
            bucket = self.buckets[client_ip]
            return {
                "remaining": bucket["tokens"],
                "requests_this_minute": bucket["requests_this_minute"],
            }


# ─────────────────────────────────────────────────────────────────────────────
# Input Validation
# ─────────────────────────────────────────────────────────────────────────────

class InputValidator:
    """Validates user input before processing."""

    @staticmethod
    def validate_question(question: str) -> tuple[bool, Optional[str]]:
        """
        Validate question input.
        Returns: (is_valid, error_message)
        """
        if not question or not isinstance(question, str):
            return False, "Question must be a non-empty string."

        question = question.strip()

        # Length checks
        if len(question) < settings.min_question_length:
            return False, f"Question too short (min {settings.min_question_length} characters)."
        if len(question) > settings.max_question_length:
            return False, f"Question too long (max {settings.max_question_length} characters)."

        # Banned keywords (configurable via BANNED_KEYWORDS env var)
        if settings.banned_keywords:
            q_lower = question.lower()
            for keyword in settings.banned_keywords:
                if keyword.strip().lower() in q_lower:
                    return False, f"Question contains prohibited keyword: '{keyword}'."

        return True, None

    @staticmethod
    def validate_portfolio_code(portfolio_code: str) -> tuple[bool, Optional[str]]:
        """Validate portfolio code format."""
        if not portfolio_code or not isinstance(portfolio_code, str):
            return False, "Portfolio code must be a non-empty string."

        # Allow alphanumeric, dash, underscore
        if not re.match(r"^[A-Z0-9\-_]{1,20}$", portfolio_code):
            return False, "Invalid portfolio code format."

        return True, None


# ─────────────────────────────────────────────────────────────────────────────
# Quota Tracking (in-memory, keyed by IP + date)
# ─────────────────────────────────────────────────────────────────────────────

class QuotaTracker:
    """Track daily quotas: question count, token usage."""

    def __init__(self):
        # Format: {ip: {date_str: {questions, tokens}}}
        self.quotas: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"questions": 0, "tokens": 0}))
        self.lock = asyncio.Lock()

    async def check_question_quota(self, client_ip: str) -> bool:
        """Check if IP can ask another question today."""
        async with self.lock:
            today = date.today().isoformat()
            current = self.quotas[client_ip][today]["questions"]
            return current < settings.max_questions_per_day

    async def check_token_quota(self, client_ip: str, estimated_tokens: int) -> bool:
        """Check if IP can use estimated tokens today."""
        async with self.lock:
            today = date.today().isoformat()
            current = self.quotas[client_ip][today]["tokens"]
            return (current + estimated_tokens) <= settings.max_tokens_per_day

    async def record_question(self, client_ip: str, estimated_tokens: int = 0) -> None:
        """Record a question and token usage."""
        async with self.lock:
            today = date.today().isoformat()
            self.quotas[client_ip][today]["questions"] += 1
            self.quotas[client_ip][today]["tokens"] += estimated_tokens

    async def get_daily_usage(self, client_ip: str) -> dict[str, int]:
        """Get today's usage stats for an IP."""
        async with self.lock:
            today = date.today().isoformat()
            return self.quotas[client_ip][today].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Cost Tracking (OpenAI API spend)
# ─────────────────────────────────────────────────────────────────────────────

class CostTracker:
    """Track OpenAI API spending; enforce daily limit."""

    def __init__(self):
        # Format: {date_str: total_cost_usd}
        self.daily_costs: dict[str, float] = defaultdict(float)
        self.lock = asyncio.Lock()

    async def get_today_spend(self) -> float:
        """Get total spending today."""
        async with self.lock:
            today = date.today().isoformat()
            return self.daily_costs[today]

    async def can_spend(self, estimated_cost: float) -> bool:
        """Check if we can spend estimated_cost today."""
        async with self.lock:
            today = date.today().isoformat()
            current = self.daily_costs[today]
            can_afford = (current + estimated_cost) <= settings.max_daily_spend_usd
            if not can_afford:
                log.warning(
                    f"Cost limit exceeded: current=${current:.2f}, "
                    f"request=${estimated_cost:.2f}, limit=${settings.max_daily_spend_usd:.2f}"
                )
            return can_afford

    async def record_cost(self, cost_usd: float) -> None:
        """Record actual API cost."""
        async with self.lock:
            today = date.today().isoformat()
            self.daily_costs[today] += cost_usd

    async def get_remaining_budget(self) -> float:
        """Get remaining budget for today."""
        today_spend = await self.get_today_spend()
        return max(0.0, settings.max_daily_spend_usd - today_spend)


# ─────────────────────────────────────────────────────────────────────────────
# Request Logging & Monitoring
# ─────────────────────────────────────────────────────────────────────────────

class RequestLogger:
    """Log all requests for monitoring and debugging."""

    def __init__(self):
        self.logs: list[dict[str, Any]] = []
        self.lock = asyncio.Lock()

    async def log_request(
        self,
        client_ip: str,
        endpoint: str,
        method: str,
        question: Optional[str] = None,
        response_status: int = 200,
        response_time_ms: float = 0.0,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        """Log a request with details."""
        if not settings.log_all_requests:
            return

        async with self.lock:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "client_ip": client_ip,
                "endpoint": endpoint,
                "method": method,
                "question_preview": (question[:100] + "...") if question and len(question) > 100 else question,
                "response_status": response_status,
                "response_time_ms": response_time_ms,
                "tokens_used": tokens_used,
                "cost_usd": cost_usd,
                "error": error,
            }
            self.logs.append(log_entry)

            # Keep only last 10000 logs in memory
            if len(self.logs) > 10000:
                self.logs = self.logs[-10000:]

            # Log to file/stdout
            log.info(json.dumps(log_entry))

    async def get_recent_logs(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent request logs."""
        async with self.lock:
            return self.logs[-limit:]

    async def get_logs_for_ip(self, client_ip: str, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent logs for a specific IP."""
        async with self.lock:
            return [log for log in self.logs if log["client_ip"] == client_ip][-limit:]


# ─────────────────────────────────────────────────────────────────────────────
# Global Instances
# ─────────────────────────────────────────────────────────────────────────────

rate_limiter = RateLimiter(rpm=settings.rate_limit_requests_per_minute)
quota_tracker = QuotaTracker()
cost_tracker = CostTracker()
request_logger = RequestLogger()
input_validator = InputValidator()
