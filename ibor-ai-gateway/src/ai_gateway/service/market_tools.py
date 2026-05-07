from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import yfinance as yf

from ai_gateway.config.settings import settings

log = logging.getLogger(__name__)


class MarketTools:
    """Async wrappers around yfinance for external market context.

    All methods use asyncio.to_thread (yfinance is synchronous).
    All failures are caught and returned as partial data — never raises.

    Includes in-memory cache for market data (5-minute TTL).
    """

    def __init__(self, cache_ttl_minutes: int = 5):
        """Initialize market tools with optional caching.

        Args:
            cache_ttl_minutes: Cache time-to-live in minutes (default 5)
        """
        self._cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}
        self._cache_ttl = timedelta(minutes=cache_ttl_minutes)
        log.info(f"MarketTools initialized with {cache_ttl_minutes}-minute cache")

    def _get_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from cache if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp < self._cache_ttl:
                log.debug(f"Cache hit: {key}")
                return value
            else:
                del self._cache[key]  # Expired
        return None

    def _set_cache(self, key: str, value: Dict[str, Any]) -> None:
        """Store value in cache with current timestamp."""
        self._cache[key] = (value, datetime.now())
        log.debug(f"Cache set: {key}")

    async def get_market_snapshot(self, ticker: str) -> Dict[str, Any]:
        """Current price, change%, volume, fundamentals, and analyst rating."""
        cache_key = f"snapshot:{ticker}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        result = await asyncio.to_thread(self._snapshot_sync, ticker)
        self._set_cache(cache_key, result)
        return result

    async def get_news(self, ticker: str) -> Dict[str, Any]:
        """Last 5 news headlines for a ticker."""
        cache_key = f"news:{ticker}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        result = await asyncio.to_thread(self._news_sync, ticker)
        self._set_cache(cache_key, result)
        return result

    async def get_earnings(self, ticker: str) -> Dict[str, Any]:
        """Next earnings date + EPS estimates."""
        cache_key = f"earnings:{ticker}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        result = await asyncio.to_thread(self._earnings_sync, ticker)
        self._set_cache(cache_key, result)
        return result

    async def get_macro_snapshot(self) -> Dict[str, Any]:
        """S&P 500, VIX, and 10Y yield — always fetched for portfolio-level questions."""
        cache_key = "macro:global"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        result = await asyncio.to_thread(self._macro_sync)
        self._set_cache(cache_key, result)
        return result

    # --- synchronous implementations (run in thread pool) ---

    def _snapshot_sync(self, ticker: str) -> Dict[str, Any]:
        try:
            info = yf.Ticker(ticker).info or {}
            return {
                "ticker": ticker,
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "change_pct": info.get("regularMarketChangePercent"),
                "volume": info.get("regularMarketVolume"),
                "avg_volume_30d": info.get("averageVolume"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "week52_high": info.get("fiftyTwoWeekHigh"),
                "week52_low": info.get("fiftyTwoWeekLow"),
                "analyst_target_price": info.get("targetMeanPrice"),
                "analyst_rating": info.get("recommendationKey"),
            }
        except Exception as exc:
            log.warning("snapshot failed %s: %s", ticker, exc)
            return {"ticker": ticker, "error": str(exc)}

    def _news_sync(self, ticker: str) -> Dict[str, Any]:
        try:
            raw_news = yf.Ticker(ticker).news or []
            headlines = []
            for item in raw_news[:5]:
                # yfinance >= 0.2.50 nests content under "content" key
                content = item.get("content", item)
                provider = content.get("provider", {})
                headlines.append({
                    "title": content.get("title", ""),
                    "published": content.get("pubDate", content.get("providerPublishTime", "")),
                    "source": provider.get("displayName", "") if isinstance(provider, dict) else "",
                })
            return {"ticker": ticker, "headlines": headlines}
        except Exception as exc:
            log.warning("news failed %s: %s", ticker, exc)
            return {"ticker": ticker, "headlines": [], "error": str(exc)}

    def _earnings_sync(self, ticker: str) -> Dict[str, Any]:
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            earnings_date: Optional[str] = None
            try:
                cal = t.calendar
                if cal is not None:
                    if isinstance(cal, dict):
                        ed = cal.get("Earnings Date")
                        if ed:
                            earnings_date = str(ed[0]) if isinstance(ed, list) else str(ed)
                    elif hasattr(cal, "loc") and "Earnings Date" in cal.index:
                        ed = cal.loc["Earnings Date"]
                        earnings_date = str(ed.iloc[0]) if hasattr(ed, "iloc") else str(ed)
            except Exception:
                pass
            return {
                "ticker": ticker,
                "next_earnings_date": earnings_date,
                "eps_estimate_forward": info.get("forwardEps"),
                "eps_trailing": info.get("trailingEps"),
            }
        except Exception as exc:
            log.warning("earnings failed %s: %s", ticker, exc)
            return {"ticker": ticker, "error": str(exc)}

    def _macro_sync(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for name, sym in settings.macro_tickers.items():
            try:
                info = yf.Ticker(sym).info or {}
                result[name] = info.get("regularMarketPrice") or info.get("currentPrice")
            except Exception as exc:
                log.warning("macro %s failed: %s", sym, exc)
                result[name] = None
        return result