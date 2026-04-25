from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from ai_gateway.repository.ibor_repository import IborRepository
from ai_gateway.model.schemas import IborAnswer


class IborService:
    """Fetches data from IborRepository and returns clean IborAnswer envelopes.

    Returns deterministic numbers only — no narrative summaries.
    Remaps Spring Boot field names to clean analyst-facing names.
    summary is always None; LlmService fills it in for chat requests.

    Includes a simple in-memory cache for historical (immutable) responses.
    """

    def __init__(self, client: IborRepository) -> None:
        self._client = client
        self._cache: Dict[Tuple, IborAnswer] = {}

    def _cached(self, key: Tuple) -> Optional[IborAnswer]:
        return self._cache.get(key)

    def _store(self, key: Tuple, answer: IborAnswer) -> None:
        self._cache[key] = answer

    async def positions(
        self,
        portfolio_code: str,
        as_of: date,
        account_code: Optional[str] = None,
        base_currency: Optional[str] = None,
        source: Optional[str] = None,
        page: int = 0,
        size: int = 500,
    ) -> IborAnswer:
        cache_key = ("positions", portfolio_code, as_of.isoformat(), account_code, base_currency, source, page, size)
        if cached := self._cached(cache_key):
            return cached

        raw = await self._client.get_positions(
            portfolio_code=portfolio_code,
            as_of=as_of.isoformat(),
            account_code=account_code,
            base_currency=base_currency,
            source=source,
            page=page,
            size=size,
        )
        items = _to_list(raw)
        positions = [_remap_position(p) for p in items]
        total_mv = _sum_decimal(p.get("marketValue") for p in positions)
        currencies = sorted({p["currency"] for p in positions if p.get("currency")})

        qs = f"portfolioCode={portfolio_code}&asOf={as_of.isoformat()}"
        if account_code:
            qs += f"&accountCode={account_code}"
        answer = IborAnswer(
            question=f"positions({portfolio_code}, {as_of})",
            as_of=as_of,
            data={
                "positions": positions,
                "totalMarketValue": float(total_mv),
                "count": len(positions),
                "currency": currencies[0] if len(currencies) == 1 else "/".join(currencies),
            },
            source=f"{self._client.base}/positions?{qs}",
        )
        self._store(cache_key, answer)
        return answer

    async def trades(
        self,
        portfolio_code: str,
        instrument_code: str,
        as_of: date,
        page: int = 0,
        size: int = 500,
    ) -> IborAnswer:
        cache_key = ("trades", portfolio_code, instrument_code, as_of.isoformat(), page, size)
        if cached := self._cached(cache_key):
            return cached

        raw = await self._client.get_position_drilldown(
            portfolio_code=portfolio_code,
            instrument_code=instrument_code,
            as_of=as_of.isoformat(),
            page=page,
            size=size,
        )
        parent = raw if isinstance(raw, dict) else {}
        raw_txns = _to_list(parent.get("transactions", raw))
        transactions = [_remap_transaction(t, parent) for t in raw_txns]
        net_qty = _sum_decimal(t.get("quantity") for t in transactions)
        gross = _sum_decimal(_mul(t.get("quantity"), t.get("price")) for t in transactions)

        rel_path = f"/positions/{portfolio_code}/{instrument_code}?asOf={as_of.isoformat()}"
        answer = IborAnswer(
            question=f"trades({portfolio_code}, {instrument_code}, {as_of})",
            as_of=as_of,
            data={
                "transactions": transactions,
                "netQty": float(net_qty),
                "grossAmount": float(gross),
                "count": len(transactions),
            },
            source=f"{self._client.base}{rel_path}",
        )
        self._store(cache_key, answer)
        return answer

    async def prices(
        self,
        instrument_code: str,
        from_date: date,
        to_date: date,
        source: Optional[str] = None,
        base_currency: Optional[str] = None,
        page: int = 0,
        size: int = 500,
    ) -> IborAnswer:
        cache_key = ("prices", instrument_code, from_date.isoformat(), to_date.isoformat(), source, base_currency, page, size)
        if cached := self._cached(cache_key):
            return cached

        raw = await self._client.get_prices(
            instrument_code=instrument_code,
            from_date=from_date.isoformat(),
            to_date=to_date.isoformat(),
            source=source,
            base_currency=base_currency,
            page=page,
            size=size,
        )
        series = _to_list(raw)
        prices = [_remap_price(p) for p in series]
        values = [v for p in prices if (v := _as_decimal(p.get("price"))) is not None]
        ccy = prices[-1].get("currency") if prices else None

        qs = f"from={from_date.isoformat()}&to={to_date.isoformat()}"
        answer = IborAnswer(
            question=f"prices({instrument_code}, {from_date}, {to_date})",
            as_of=to_date,
            data={
                "prices": prices,
                "count": len(values),
                "min": float(min(values)) if values else None,
                "max": float(max(values)) if values else None,
                "last": float(values[-1]) if values else None,
                "currency": ccy,
            },
            source=f"{self._client.base}/prices/{instrument_code}?{qs}",
        )
        self._store(cache_key, answer)
        return answer

    async def pnl(
        self,
        portfolio_code: str,
        as_of: date,
        prior: date,
        instrument_code: Optional[str] = None,
    ) -> IborAnswer:
        curr = await self.positions(portfolio_code=portfolio_code, as_of=as_of)
        prev = await self.positions(portfolio_code=portfolio_code, as_of=prior)

        curr_items = curr.data.get("positions", [])
        prev_items = prev.data.get("positions", [])

        if instrument_code:
            curr_items = [p for p in curr_items if p.get("instrument") == instrument_code]
            prev_items = [p for p in prev_items if p.get("instrument") == instrument_code]

        curr_mv = _sum_decimal(p.get("marketValue") for p in curr_items)
        prev_mv = _sum_decimal(p.get("marketValue") for p in prev_items)
        delta = curr_mv - prev_mv

        return IborAnswer(
            question=f"pnl({portfolio_code}, {as_of}, prior={prior})",
            as_of=as_of,
            data={
                "portfolio": portfolio_code,
                "scope": instrument_code or "PORTFOLIO",
                "asOf": str(as_of),
                "prior": str(prior),
                "currentMarketValue": float(curr_mv),
                "previousMarketValue": float(prev_mv),
                "delta": float(delta),
            },
            source=curr.source,
        )


# --- Field remappers ---

def _remap_position(p: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "instrument": p.get("instrumentId"),
        "name": p.get("instrumentName"),
        "type": p.get("instrumentType"),
        "quantity": p.get("netQty"),
        "price": p.get("price"),
        "marketValue": p.get("mktValue"),
        "currency": p.get("currency"),
        "snapDate": p.get("snapDate"),
    }


def _remap_transaction(t: Dict[str, Any], parent: Dict[str, Any] = {}) -> Dict[str, Any]:
    return {
        "tradeId": t.get("externalId"),
        "instrument": parent.get("instrumentCode"),
        "name": parent.get("instrumentName"),
        "type": parent.get("instrumentType"),
        "date": t.get("transactionDate"),
        "action": t.get("action"),
        "quantity": t.get("quantity"),
        "price": t.get("price"),
        "grossAmount": t.get("grossAmount"),
        "broker": t.get("broker"),
    }


def _remap_price(p: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "date": p.get("priceTs"),
        "price": p.get("price"),
        "currency": p.get("currency"),
    }


# --- Decimal helpers ---

def _to_list(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]
    if isinstance(raw, dict):
        for key in ("positions", "transactions", "prices"):
            if isinstance(raw.get(key), list):
                return [r for r in raw[key] if isinstance(r, dict)]
    return []


def _as_decimal(x: Any) -> Optional[Decimal]:
    if x is None:
        return None
    try:
        return Decimal(str(x))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _sum_decimal(iterable) -> Decimal:
    return sum(
        (d for x in iterable if (d := _as_decimal(x)) is not None),
        Decimal("0"),
    )


def _mul(a: Any, b: Any) -> Optional[Decimal]:
    da, db = _as_decimal(a), _as_decimal(b)
    return da * db if da is not None and db is not None else None
