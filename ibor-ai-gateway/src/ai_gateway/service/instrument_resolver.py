from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from ai_gateway.config.db import PgPool

log = logging.getLogger(__name__)

_STAGE1_SQL = """
SELECT di.instrument_code, di.instrument_name, di.instrument_type
FROM ibor.dim_instrument di
LEFT JOIN ibor.dim_instrument_equity eq USING (instrument_vid)
WHERE di.is_current = true
  AND (lower(eq.ticker) = lower(%(q)s) OR lower(di.instrument_code) = lower(%(q)s))
ORDER BY di.instrument_type
"""

_STAGE2_SQL = """
SELECT instrument_code, instrument_name, instrument_type,
       similarity(instrument_name, %(q)s) AS score
FROM ibor.dim_instrument
WHERE is_current = true
  AND similarity(instrument_name, %(q)s) > 0.25
ORDER BY score DESC
LIMIT 8
"""


@dataclass
class InstrumentMatch:
    code: str
    name: str
    instrument_type: str


@dataclass
class ResolveResult:
    matches: List[InstrumentMatch] = field(default_factory=list)
    clarification: Optional[str] = None

    @property
    def resolved(self) -> Optional[str]:
        """Returns the single canonical code, or None if ambiguous/not found."""
        return self.matches[0].code if len(self.matches) == 1 else None

    @property
    def is_ambiguous(self) -> bool:
        return self.clarification is not None


class InstrumentResolver:
    """Two-stage instrument resolution: exact ticker/code first, pg_trgm name fuzzy second.

    Stage 1 — indexed lookup: ticker (equities) or instrument_code exact match.
    Stage 2 — GIN trigram search on instrument_name (only if Stage 1 finds nothing).
    Multiple results of mixed types → clarifying question returned to the user.
    """

    def __init__(self, pg_pool: PgPool) -> None:
        self._pool = pg_pool

    def resolve(self, query: str, type_hint: Optional[str] = None) -> ResolveResult:
        """Resolve a user-supplied name/ticker/code to canonical instrument_code(s)."""
        query = query.strip()
        if not query:
            return ResolveResult()

        # Stage 1: exact ticker or instrument_code match
        rows = self._pool.fetch_all(_STAGE1_SQL, {"q": query})
        stage1_name: Optional[str] = None  # company name from Stage 1, used if we need Stage 1.5
        if rows:
            matches = [InstrumentMatch(r["instrument_code"], r["instrument_name"], r["instrument_type"]) for r in rows]
            if type_hint:
                filtered = [m for m in matches if m.instrument_type == type_hint.upper()]
                if filtered:
                    return self._evaluate(query, filtered, stage="exact")
                # Stage 1 hit is the wrong type — save company name for Stage 1.5
                stage1_name = matches[0].name
            else:
                return self._evaluate(query, matches, stage="exact")

        # Stage 2: fuzzy name match via pg_trgm
        # If Stage 1 found a company of the wrong type, search by company name (not raw query)
        name_query = stage1_name if stage1_name else query
        rows = self._pool.fetch_all(_STAGE2_SQL, {"q": name_query})
        matches = [InstrumentMatch(r["instrument_code"], r["instrument_name"], r["instrument_type"]) for r in rows]
        if type_hint and matches:
            filtered = [m for m in matches if m.instrument_type == type_hint.upper()]
            if filtered:
                matches = filtered
        return self._evaluate(query, matches, stage="fuzzy")

    def _evaluate(self, query: str, matches: List[InstrumentMatch], stage: str) -> ResolveResult:
        if not matches:
            return ResolveResult(
                clarification=f'I don\'t recognise "{query}" as an instrument in the portfolio. '
                              f'Try using a ticker (e.g. AAPL), instrument code (e.g. EQ-AAPL), or full name.'
            )

        if len(matches) == 1:
            log.debug("resolved %r → %s (%s)", query, matches[0].code, stage)
            return ResolveResult(matches=matches)

        # Multiple matches — check if they're all the same type
        types = {m.instrument_type for m in matches}
        if len(types) == 1:
            # Same type (e.g. multiple options) — return all, no clarification needed
            log.debug("resolved %r → %d matches, same type %s", query, len(matches), types)
            return ResolveResult(matches=matches)

        # Mixed types — ask the user to clarify
        options = "\n".join(
            f"  {i+1}. {m.code} — {m.name} ({m.instrument_type.lower()})"
            for i, m in enumerate(matches)
        )
        clarification = (
            f'I found {len(matches)} instruments matching "{query}":\n{options}\n'
            f'Which one did you mean?'
        )
        log.debug("ambiguous %r → %d matches across types %s", query, len(matches), types)
        return ResolveResult(matches=matches, clarification=clarification)
