from __future__ import annotations

import logging
from typing import Optional, Tuple

from groq import AsyncGroq

log = logging.getLogger(__name__)

# Llama Guard 3 category definitions — S14 covers prompt injection / jailbreaks
_CATEGORIES = """\
S1: Violent Crimes
S2: Non-Violent Crimes
S3: Sex-Related Crimes
S4: Child Sexual Exploitation
S5: Defamation
S6: Specialized Advice
S7: Privacy
S8: Intellectual Property
S9: Indiscriminate Weapons
S10: Hate
S11: Suicide & Self-Harm
S12: Sexual Content
S13: Elections
S14: Code Interpreter Abuse"""

_CATEGORY_LABELS = {
    "s1": "violent crimes",
    "s2": "non-violent crimes",
    "s3": "sex-related crimes",
    "s4": "child exploitation",
    "s5": "defamation",
    "s6": "specialized advice",
    "s7": "privacy violation",
    "s8": "intellectual property",
    "s9": "weapons",
    "s10": "hate speech",
    "s11": "self-harm",
    "s12": "sexual content",
    "s13": "elections",
    "s14": "prompt injection or jailbreak",
}


class LlamaGuardService:
    """Pre-filter using Meta's Llama Guard 3 via Groq's free inference API.

    Catches prompt injection, jailbreaks, and other unsafe content before the
    question reaches Claude. Falls through safely (is_safe=True) if GROQ_API_KEY
    is not configured or if the Groq API is unavailable — a Groq outage will not
    take down the gateway.

    Free tier: 14,400 requests/day, 30 req/min on llama-guard-3-8b.
    """

    _MODEL = "llama-guard-3-8b"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._enabled = bool(api_key)
        self._client = AsyncGroq(api_key=api_key) if api_key else None
        if self._enabled:
            log.info("LlamaGuardService enabled (model=%s)", self._MODEL)
        else:
            log.info("LlamaGuardService disabled — GROQ_API_KEY not set")

    async def check(self, question: str) -> Tuple[bool, Optional[str]]:
        """Classify a user question against Llama Guard 3's safety policy.

        Returns:
            (is_safe, reason)
            - is_safe=True  → question is safe to proceed
            - is_safe=False → question is unsafe; reason names the violated category
        """
        if not self._enabled:
            return True, None

        try:
            response = await self._client.chat.completions.create(
                model=self._MODEL,
                messages=[{"role": "user", "content": self._build_prompt(question)}],
                max_tokens=20,
                temperature=0,
            )
            return self._parse(response.choices[0].message.content, question)
        except Exception as exc:
            log.warning("LlamaGuard check failed (%s) — falling through", exc)
            return True, None

    def _parse(self, raw: str, question: str) -> Tuple[bool, Optional[str]]:
        text = raw.strip().lower()
        if not text.startswith("unsafe"):
            return True, None

        lines = text.split("\n", 1)
        category_codes = lines[1].strip() if len(lines) > 1 else ""
        # Map e.g. "s14" → "prompt injection or jailbreak"
        labels = [
            _CATEGORY_LABELS.get(code.strip(), code.strip())
            for code in category_codes.split(",")
            if code.strip()
        ]
        reason = ", ".join(labels) if labels else "unsafe content"
        log.warning("LlamaGuard blocked [%s]: %.120s", reason, question)
        return False, reason

    def _build_prompt(self, question: str) -> str:
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "Task: Check if there is unsafe content in 'User' messages in conversations "
            "according our safety policy with the below categories.\n\n"
            "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
            f"{_CATEGORIES}\n"
            "<END UNSAFE CONTENT CATEGORIES>\n\n"
            "<BEGIN CONVERSATION>\n\n"
            f"User: {question}\n\n"
            "<END CONVERSATION>\n\n"
            "Provide your safety assessment for ONLY THE LAST User message in the above conversation:\n"
            " - First line must read 'safe' or 'unsafe'.\n"
            " - If unsafe, a second line must include a comma-separated list of violated categories."
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
