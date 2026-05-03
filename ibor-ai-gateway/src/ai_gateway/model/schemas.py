from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class QuotaStatus(BaseModel):
    """Quota status for user (IP-based)."""
    questions_today: int
    questions_limit: int
    questions_remaining: int
    quota_exceeded: bool
    reset_time: Optional[datetime] = None


class IborAnswer(BaseModel):
    """Response envelope returned by all routes — REST and chat alike."""
    question: str
    as_of: Optional[date] = None
    summary: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = None
    gaps: List[str] = Field(default_factory=list)
    clarification: Optional[str] = None
    quota_status: Optional[QuotaStatus] = None


# --- Request models ---

class PositionsRequest(BaseModel):
    portfolio_code: str
    as_of: date
    base_currency: Optional[str] = None
    source: Optional[str] = None
    page: int = 0
    size: int = 500


class TradesRequest(BaseModel):
    portfolio_code: str
    instrument_code: str
    as_of: date
    page: int = 0
    size: int = 500


class PricesRequest(BaseModel):
    instrument_code: str
    from_date: date
    to_date: date
    source: Optional[str] = None
    base_currency: Optional[str] = None
    page: int = 0
    size: int = 500


class PnLRequest(BaseModel):
    portfolio_code: str
    as_of: date
    prior: date
    instrument_code: Optional[str] = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=10, max_length=2000)
    portfolio_code: Optional[str] = Field(default=None, pattern=r"^[A-Z0-9\-_]{1,20}$")
    as_of: Optional[date] = None
    market_contents: Optional[bool] = True

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        return v.strip()
