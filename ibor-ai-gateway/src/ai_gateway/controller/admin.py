from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Query
from psycopg.rows import dict_row

log = logging.getLogger(__name__)

# Injected from main.py at startup (same pattern as conversation_test.py)
pg_pool = None

router = APIRouter(prefix="/admin", tags=["Admin"])

# ── IP geolocation ────────────────────────────────────────────────────────────

_geo_cache: Dict[str, Dict] = {}   # in-process cache; survives for life of the process


def _geolocate_ips(ips: List[str]) -> Dict[str, Dict]:
    """Batch-geolocate IPs via ip-api.com (free, no key, up to 100 IPs per call).
    Results are cached in-process. Private/loopback IPs return a placeholder.
    """
    PRIVATE = {"127.0.0.1", "::1", "localhost", "unknown"}
    to_fetch = [ip for ip in ips if ip not in _geo_cache and ip not in PRIVATE]

    for ip in ips:
        if ip in PRIVATE:
            _geo_cache[ip] = {"city": "local", "country": "—", "org": "—"}

    if to_fetch:
        try:
            payload = [{"query": ip, "fields": "query,country,countryCode,city,org,status"} for ip in to_fetch]
            resp = httpx.post(
                "http://ip-api.com/batch",
                json=payload,
                timeout=5.0,
            )
            if resp.status_code == 200:
                for entry in resp.json():
                    ip = entry.get("query", "")
                    if entry.get("status") == "success":
                        _geo_cache[ip] = {
                            "city":         entry.get("city"),
                            "country":      entry.get("country"),
                            "country_code": entry.get("countryCode"),
                            "org":          entry.get("org"),
                        }
                    else:
                        _geo_cache[ip] = {"city": None, "country": None, "org": None}
        except Exception as e:
            log.warning("IP geolocation failed: %s", e)
            for ip in to_fetch:
                if ip not in _geo_cache:
                    _geo_cache[ip] = {"city": None, "country": None, "org": None}

    return {ip: _geo_cache.get(ip, {}) for ip in ips}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_messages(raw) -> List[Dict]:
    if not raw:
        return []
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _fmt(dt) -> Optional[str]:
    return dt.isoformat() if dt else None


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.get("/conversations")
def get_conversations(
    analyst: Optional[str] = Query(None, description="Filter by analyst_id (IP)"),
    limit: int = Query(20, ge=1, le=200, description="Max conversations (default 20)"),
    questions_only: bool = Query(False, description="Return flat question/answer list"),
    geolocate: bool = Query(True, description="Enrich IPs with location (ip-api.com)"),
) -> Dict[str, Any]:
    """
    Admin view: who used the AI chatbot, what did they ask, and what did we say?

    Protected by X-API-Key enforced in SecurityMiddleware.

    curl examples:
        # Full report
        curl -H "X-API-Key: $API_KEY" https://<gateway>/admin/conversations | jq .

        # Flat Q&A list — easiest to skim
        curl -H "X-API-Key: $API_KEY" "https://<gateway>/admin/conversations?questions_only=true" | jq .

        # Filter to one IP
        curl -H "X-API-Key: $API_KEY" "https://<gateway>/admin/conversations?analyst=173.3.239.214" | jq .

        # Skip geolocation
        curl -H "X-API-Key: $API_KEY" "https://<gateway>/admin/conversations?geolocate=false" | jq .
    """
    with pg_pool.connection() as conn:
        conn.row_factory = dict_row

        # ── Overall summary ──────────────────────────────────────────────
        s = conn.execute("""
            SELECT
                COUNT(DISTINCT analyst_id)      AS unique_analysts,
                COUNT(DISTINCT session_id)      AS total_sessions,
                COUNT(*)                        AS total_conversations,
                COALESCE(SUM(message_count), 0) AS total_messages,
                MIN(created_at)                 AS first_conversation,
                MAX(updated_at)                 AS last_activity
            FROM conv.conversation
        """).fetchone()

        summary = {
            "unique_analysts":     s["unique_analysts"],
            "total_sessions":      s["total_sessions"],
            "total_conversations": s["total_conversations"],
            "total_messages":      s["total_messages"],
            "first_conversation":  _fmt(s["first_conversation"]),
            "last_activity":       _fmt(s["last_activity"]),
        }

        if s["total_conversations"] == 0:
            return {"summary": summary, "analysts": [], "conversations": []}

        # ── Per-analyst breakdown (with geolocation) ─────────────────────
        analyst_rows = conn.execute("""
            SELECT
                analyst_id,
                COUNT(DISTINCT session_id)      AS sessions,
                COUNT(*)                        AS conversations,
                COALESCE(SUM(message_count), 0) AS messages,
                MIN(created_at)                 AS first_seen,
                MAX(updated_at)                 AS last_seen
            FROM conv.conversation
            GROUP BY analyst_id
            ORDER BY last_seen DESC
        """).fetchall()

        unique_ips = [r["analyst_id"] for r in analyst_rows]
        geo = _geolocate_ips(unique_ips) if geolocate else {}

        analysts = []
        for r in analyst_rows:
            ip = r["analyst_id"]
            entry = {
                "analyst_id":    ip,
                "sessions":      r["sessions"],
                "conversations": r["conversations"],
                "messages":      r["messages"],
                "first_seen":    _fmt(r["first_seen"]),
                "last_seen":     _fmt(r["last_seen"]),
            }
            if geo:
                entry["location"] = geo.get(ip, {})
            analysts.append(entry)

        # ── Conversation + message detail ─────────────────────────────────
        where = "WHERE analyst_id = %s" if analyst else ""
        params = [analyst] if analyst else []

        conv_rows = conn.execute(
            f"""
            SELECT
                conversation_id, analyst_id, session_id,
                context_type, context_id, title,
                message_count, created_at, updated_at, messages
            FROM conv.conversation
            {where}
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            params + [limit],
        ).fetchall()

        # ── questions_only mode: flat Q&A pairs ───────────────────────────
        if questions_only:
            qa_list = []
            for row in conv_rows:
                msgs = _parse_messages(row["messages"])
                ip = row["analyst_id"]
                location = geo.get(ip, {}) if geo else {}
                # Pair up questions and answers in order
                for i, msg in enumerate(msgs):
                    if msg.get("role") == "analyst":
                        question = msg.get("content", "").strip()
                        # Look for the immediately following AI message
                        answer = None
                        if i + 1 < len(msgs) and msgs[i + 1].get("role") == "ai":
                            answer = msgs[i + 1].get("content", "").strip()
                        entry = {
                            "analyst_id": ip,
                            "asked_at":   _fmt(row["updated_at"]),
                            "question":   question,
                            "answer":     answer,
                        }
                        if location:
                            entry["location"] = location
                        qa_list.append(entry)
            return {"summary": summary, "qa": qa_list}

        # ── Full conversation view ────────────────────────────────────────
        conversations = []
        for row in conv_rows:
            msgs = _parse_messages(row["messages"])
            ip = row["analyst_id"]
            location = geo.get(ip, {}) if geo else {}

            exchanges = []
            for i, msg in enumerate(msgs):
                role = msg.get("role", "")
                content = msg.get("content", "").strip()
                if role == "analyst":
                    answer = None
                    if i + 1 < len(msgs) and msgs[i + 1].get("role") == "ai":
                        answer = msgs[i + 1].get("content", "").strip()
                    exchanges.append({"question": content, "answer": answer})

            entry = {
                "conversation_id": str(row["conversation_id"]),
                "analyst_id":      ip,
                "session_id":      str(row["session_id"]),
                "context_type":    row["context_type"],
                "context_id":      row["context_id"],
                "title":           row["title"],
                "message_count":   row["message_count"],
                "created_at":      _fmt(row["created_at"]),
                "updated_at":      _fmt(row["updated_at"]),
                "exchanges":       exchanges,
            }
            if location:
                entry["location"] = location
            conversations.append(entry)

        return {
            "summary":       summary,
            "analysts":      analysts,
            "conversations": conversations,
        }
