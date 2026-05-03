from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load secrets from repo root .env (never committed)
load_dotenv(Path(__file__).resolve().parents[4] / ".env")


class Settings:
    """All gateway configuration in one place.

    Defaults are hardcoded below. Override any value via environment variable or .env.
    Secrets (ANTHROPIC_API_KEY, PG_DSN) must be provided — no hardcoded defaults.
    """

    # ── Secrets (required) ───────────────────────────────────────────────────
    anthropic_api_key: str  = os.getenv("ANTHROPIC_API_KEY", "")
    pg_dsn: str             = os.getenv("PG_DSN", "postgresql://ibor:ibor@localhost:5432/ibor")
    groq_api_key: str       = os.getenv("GROQ_API_KEY", "")

    # ── LLM ─────────────────────────────────────────────────────────────────
    anthropic_model: str    = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    # ── Spring Boot middleware ───────────────────────────────────────────────
    structured_api_base: str = os.getenv("STRUCTURED_API_BASE", "http://localhost:8080/api")
    verify_ssl: bool          = os.getenv("VERIFY_SSL", "false").lower() == "true"

    # ── CORS ─────────────────────────────────────────────────────────────────
    allowed_origins: list[str] = (
        os.getenv("ALLOWED_ORIGINS", "").split(",")
        if os.getenv("ALLOWED_ORIGINS")
        else ["http://localhost:4200", "http://localhost:5173", "localhost", "127.0.0.1"]
    )

    # ── Security ─────────────────────────────────────────────────────────────
    environment: str                    = os.getenv("ENVIRONMENT", "development")
    api_key: str                        = os.getenv("API_KEY", "")
    rate_limit_requests_per_minute: int = int(os.getenv("RATE_LIMIT_RPM", "30"))
    rate_limit_enabled: bool            = os.getenv("RATE_LIMIT_ENABLED", "false").lower() == "true"
    email_whitelist_enabled: bool       = os.getenv("EMAIL_WHITELIST_ENABLED", "false").lower() == "true"
    email_whitelist: list[str]          = os.getenv("EMAIL_WHITELIST", "").split(",") if os.getenv("EMAIL_WHITELIST") else []
    max_questions_per_day: int          = int(os.getenv("MAX_QUESTIONS_PER_DAY", "100"))
    max_tokens_per_day: int             = int(os.getenv("MAX_TOKENS_PER_DAY", "500000"))
    max_daily_spend_usd: float          = float(os.getenv("MAX_DAILY_SPEND_USD", "50.0"))
    cost_tracking_enabled: bool         = os.getenv("COST_TRACKING_ENABLED", "false").lower() == "true"
    max_question_length: int            = int(os.getenv("MAX_QUESTION_LENGTH", "2000"))
    min_question_length: int            = int(os.getenv("MIN_QUESTION_LENGTH", "10"))
    banned_keywords: list[str]          = os.getenv("BANNED_KEYWORDS", "").split(",") if os.getenv("BANNED_KEYWORDS") else []
    log_all_requests: bool              = os.getenv("LOG_ALL_REQUESTS", "true").lower() == "true"
    alert_on_quota_violation: bool      = os.getenv("ALERT_ON_QUOTA_VIOLATION", "true").lower() == "true"


settings = Settings()
