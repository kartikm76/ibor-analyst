from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from ai_gateway.config.settings import settings


class IborRepository:
    """HTTP repository for the Spring Boot IBOR service.

    One persistent connection pool shared for the lifetime of the process.
    Create once at startup, call aclose() on shutdown.
    """

    def __init__(self, base_url: Optional[str] = None) -> None:
        self._base = (base_url or settings.structured_api_base).rstrip("/")
        self._http = httpx.AsyncClient(
            timeout=15.0,
            verify=settings.verify_ssl,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            headers={"X-API-Key": settings.api_key} if settings.api_key else {},
        )

    @property
    def base(self) -> str:
        return self._base

    async def aclose(self) -> None:
        await self._http.aclose()

    async def get_positions(
        self,
        portfolio_code: str,
        as_of: str,
        account_code: Optional[str] = None,
        base_currency: Optional[str] = None,
        source: Optional[str] = None,
        page: int = 0,
        size: int = 500,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"portfolioCode": portfolio_code, "asOf": as_of, "page": page, "size": size}
        if account_code:
            params["accountCode"] = account_code
        if base_currency:
            params["baseCurrency"] = base_currency
        if source:
            params["source"] = source
        r = await self._http.get(f"{self._base}/positions", params=params)
        r.raise_for_status()
        return r.json()

    async def get_position_drilldown(
        self,
        portfolio_code: str,
        instrument_code: str,
        as_of: str,
        page: int = 0,
        size: int = 500,
    ) -> Any:
        params: Dict[str, Any] = {"asOf": as_of, "page": page, "size": size}
        r = await self._http.get(
            f"{self._base}/positions/{portfolio_code}/{instrument_code}",
            params=params,
        )
        r.raise_for_status()
        return r.json()

    async def get_prices(
        self,
        instrument_code: str,
        from_date: str,
        to_date: str,
        source: Optional[str] = None,
        base_currency: Optional[str] = None,
        page: int = 0,
        size: int = 500,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"from": from_date, "to": to_date, "page": page, "size": size}
        if source:
            params["source"] = source
        if base_currency:
            params["baseCurrency"] = base_currency
        r = await self._http.get(f"{self._base}/prices/{instrument_code}", params=params)
        r.raise_for_status()
        return r.json()
