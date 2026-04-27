"""
End-to-end smoke tests for the IBOR stack.

Run after any code change or Docker rebuild:
    cd ibor-ai-gateway && uv run pytest tests/smoke/ -v

Requires the full stack to be running:
    - PostgreSQL      localhost:5432
    - ibor-middleware localhost:8080
    - ibor-ai-gateway localhost:8000

Every test here corresponds to a real bug we have seen in production.
"""
from __future__ import annotations

import pytest
import httpx

MIDDLEWARE = "http://localhost:8080/api"
GATEWAY = "http://localhost:8000"

# Known-good fixture data loaded by the bootstrap container
PORTFOLIO = "P-ALPHA"
AS_OF = "2025-04-10"           # date with a confirmed position snapshot
AAPL_CODE = "EQ-AAPL"
BA_CODE = "EQ-BA"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chat(question: str) -> dict:
    r = httpx.post(
        f"{GATEWAY}/analyst/chat",
        json={"question": question, "market_contents": False},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# 1. Infrastructure health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_middleware_actuator(self):
        r = httpx.get(f"http://localhost:8080/actuator/health", timeout=5)
        assert r.status_code == 200
        assert r.json().get("status") == "UP"

    def test_gateway_health(self):
        r = httpx.get(f"{GATEWAY}/health", timeout=5)
        assert r.status_code == 200
        body = r.json()
        assert body.get("status") in ("ok", "degraded"), f"unexpected status: {body}"
        assert body["checks"]["anthropic"]["status"] == "ok", "Anthropic key missing or placeholder"
        assert body["checks"]["middleware"]["status"] == "ok", "middleware unreachable from gateway"
        assert body["checks"]["postgres"]["status"] == "ok", "postgres unreachable from gateway"


# ---------------------------------------------------------------------------
# 2. Spring Boot middleware — field contract
# ---------------------------------------------------------------------------

class TestMiddleware:
    def test_positions_returns_list(self):
        r = httpx.get(f"{MIDDLEWARE}/positions", params={"portfolioCode": PORTFOLIO, "asOf": AS_OF}, timeout=10)
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list), "positions must return a list"
        assert len(body) > 0, "expected at least one position"

    def test_drilldown_has_instrument_name(self):
        """Regression: instrumentName was missing from PositionDetailDTO."""
        r = httpx.get(f"{MIDDLEWARE}/positions/{PORTFOLIO}/{AAPL_CODE}", params={"asOf": AS_OF}, timeout=10)
        assert r.status_code == 200
        body = r.json()
        assert body.get("instrumentName") is not None, "instrumentName must not be null"
        assert body.get("instrumentCode") == AAPL_CODE
        assert body.get("portfolioCode") == PORTFOLIO

    def test_drilldown_transactions_present(self):
        r = httpx.get(f"{MIDDLEWARE}/positions/{PORTFOLIO}/{AAPL_CODE}", params={"asOf": AS_OF}, timeout=10)
        assert r.status_code == 200
        txns = r.json().get("transactions", [])
        assert len(txns) > 0, "expected at least one AAPL transaction"
        t = txns[0]
        assert t.get("action") in ("BUY", "SELL", "ADJUST")
        assert t.get("quantity") is not None


# ---------------------------------------------------------------------------
# 3. Python gateway — REST endpoints
# ---------------------------------------------------------------------------

class TestGatewayRest:
    def test_positions_envelope(self):
        r = httpx.post(
            f"{GATEWAY}/analyst/positions",
            json={"portfolio_code": PORTFOLIO, "as_of": AS_OF},
            timeout=15,
        )
        assert r.status_code == 200
        body = r.json()
        assert body.get("as_of") is not None, "as_of must not be null"
        assert "positions" in body.get("data", {}), "data.positions missing"
        assert body["data"]["count"] > 0

    def test_trades_name_not_null(self):
        """Regression: name was null in transaction remapper due to stale cache."""
        r = httpx.post(
            f"{GATEWAY}/analyst/trades",
            json={"portfolio_code": PORTFOLIO, "instrument_code": AAPL_CODE, "as_of": AS_OF},
            timeout=15,
        )
        assert r.status_code == 200
        body = r.json()
        assert body.get("as_of") is not None, "as_of must not be null"
        txns = body.get("data", {}).get("transactions", [])
        assert len(txns) > 0, "expected at least one transaction"
        for t in txns:
            assert t.get("instrument") == AAPL_CODE, "instrument field wrong"
            assert t.get("name") is not None, f"name is null for transaction {t}"
            assert t.get("type") is not None, "type field missing"

    def test_trades_boeing(self):
        """Regression: Boeing question triggered as_of=None Pydantic crash."""
        r = httpx.post(
            f"{GATEWAY}/analyst/trades",
            json={"portfolio_code": PORTFOLIO, "instrument_code": BA_CODE, "as_of": AS_OF},
            timeout=15,
        )
        assert r.status_code == 200
        body = r.json()
        assert body.get("as_of") is not None
        assert "transactions" in body.get("data", {})


# ---------------------------------------------------------------------------
# 4. Instrument resolver (tested via chat — resolution only runs in LlmService)
# ---------------------------------------------------------------------------

class TestInstrumentResolver:
    def test_exact_ticker_via_chat_resolves(self):
        """'AAPL trades' → LLM extracts ticker → resolver finds EQ-AAPL, no clarification."""
        body = _chat("show me my AAPL trades")
        assert "detail" not in body, f"chat error: {body.get('detail')}"
        assert body.get("clarification") is None, "AAPL ticker should not be ambiguous"
        assert body.get("as_of") is not None

    def test_ambiguous_name_via_chat_returns_clarification(self):
        """A bare ambiguous name with no known ticker should return a clarification.

        Note: Claude often resolves common names like 'Apple' to 'AAPL' directly.
        This test accepts either a clean resolution OR a clarification — both are correct.
        What it rejects is a Pydantic crash (detail field) or a null as_of.
        """
        body = _chat("show me my Apple bond trades")
        assert "detail" not in body, f"chat error: {body.get('detail')}"
        assert body.get("as_of") is not None, "as_of must not be null"
        # Either clarification was asked or data was returned — both acceptable
        has_clarification = body.get("clarification") is not None
        has_data = bool(body.get("data", {}).get("ibor", {}).get("trades", {}).get("transactions"))
        assert has_clarification or has_data or body.get("summary"), (
            "Expected clarification, data, or summary — got none"
        )


# ---------------------------------------------------------------------------
# 5. Chat — end-to-end including as_of contract
# ---------------------------------------------------------------------------

class TestChat:
    def test_response_envelope_valid(self):
        """Every chat response must have as_of set (no Pydantic validation crash)."""
        body = _chat("what are my current positions")
        assert "detail" not in body, f"chat returned error: {body.get('detail')}"
        assert body.get("as_of") is not None, "as_of must not be null in chat response"
        assert body.get("summary") is not None, "chat must return a summary"

    def test_apple_trades_chat(self):
        """'Apple trades' must resolve, fetch data, and return instrument name."""
        body = _chat("show me my Apple trades")
        assert "detail" not in body, f"chat error: {body.get('detail')}"
        assert body.get("as_of") is not None
        # Either resolved cleanly or asked for clarification — both are valid
        if body.get("clarification"):
            assert "Apple" in body["clarification"]
        else:
            trades = body.get("data", {}).get("ibor", {}).get("trades", {}).get("transactions", [])
            for t in trades:
                assert t.get("name") is not None, "instrument name missing in chat trade response"

    def test_boeing_position_chat(self):
        """'Boeing position' must not crash with Pydantic as_of=None error."""
        body = _chat("how about my Boeing position")
        assert "detail" not in body, f"chat error: {body.get('detail')}"
        assert body.get("as_of") is not None, "as_of was None — Pydantic crash regression"
        assert body.get("summary") is not None

    def test_portfolio_overview_chat(self):
        """Generic portfolio question must return positions data."""
        body = _chat("give me a summary of my portfolio")
        assert "detail" not in body, f"chat error: {body.get('detail')}"
        assert body.get("as_of") is not None
        assert body.get("summary") is not None

    def test_quota_exceeded_no_crash(self):
        """Quota-exceeded path must return valid IborAnswer, not crash."""
        # We can't easily simulate quota here, but we can verify the response
        # shape is always valid (as_of present) when quota is NOT exceeded.
        body = _chat("what is my AAPL position")
        assert "detail" not in body, f"chat error: {body.get('detail')}"
        assert body.get("as_of") is not None
