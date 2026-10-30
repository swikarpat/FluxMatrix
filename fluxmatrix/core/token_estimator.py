"""Prompt analysis, zero-overhead token estimation, and query complexity scoring.

Designed for microsecond-level execution on the gateway ingress path before
routing to backend inference workers.
"""

import re
from typing import List, Tuple
from fluxmatrix.core.models import ChatMessage


# Regex patterns associated with high cognitive load or reasoning depth
COMPLEX_PATTERNS = [
    re.compile(r"\b(prove|derive|architect|benchmark|trace|optimize|profile)\b", re.IGNORECASE),
    re.compile(r"\b(step[- ]by[- ]step|think carefully|chain of thought|reasoning)\b", re.IGNORECASE),
    re.compile(r"(```[a-zA-Z]*\n[\s\S]*?```)"),  # Code blocks
    re.compile(r"(\bdef\b|\bclass\b|\bfunction\b|\bstruct\b|\bimpl\b|\bSELECT\b|\bFROM\b)"),
    re.compile(r"(\$|\\\[|KaTeX|\bKaTeX\b|KaTeX|\bLaTeX\b|\bO\([nN0-9log\s\*\^]+\))"), # Math/complexity
]

# Regex patterns associated with simple lookup / extraction / summarization
SIMPLE_PATTERNS = [
    re.compile(r"\b(summarize|tldr|translate|rephrase|extract|correct spelling|capitalize)\b", re.IGNORECASE),
    re.compile(r"\b(yes or no|true or false|who is|what is the capital|when was)\b", re.IGNORECASE),
]


class PromptAnalyzer:
    """Fast, zero-GPU prompt analysis and complexity scoring engine."""

    def __init__(self, avg_chars_per_token: float = 3.8):
        self.avg_chars_per_token = avg_chars_per_token

    def estimate_tokens(self, messages: List[ChatMessage]) -> int:
        """Estimate token count across a multi-turn conversation quickly without full BPE overhead.

        Average English / code token ratio is ~3.6 - 4.0 characters per token plus formatting overhead.
        """
        total_chars = 0
        message_overhead = 4  # tokens per message for role delimiters

        for msg in messages:
            content = msg.content or ""
            total_chars += len(content)

        raw_tokens = int(total_chars / self.avg_chars_per_token)
        return max(1, raw_tokens + (len(messages) * message_overhead))

    def compute_complexity_score(self, messages: List[ChatMessage]) -> Tuple[float, List[str]]:
        """Calculate a normalized complexity score between 0.0 (simplest) and 1.0 (most complex).

        Returns:
            Tuple of (score, signals_detected)
        """
        if not messages:
            return 0.0, ["empty_prompt"]

        signals: List[str] = []
        score = 0.20  # Baseline conversational score

        # Aggregate user & system texts
        full_text = " ".join([m.content or "" for m in messages])
        token_count = self.estimate_tokens(messages)

        # 1. Length Factor
        if token_count > 3000:
            score += 0.35
            signals.append("long_context_>3k")
        elif token_count > 1000:
            score += 0.20
            signals.append("medium_context_>1k")

        # 2. Multi-turn conversation depth
        if len(messages) > 6:
            score += 0.15
            signals.append("deep_conversation_turn")

        # 3. Code & Reasoning Pattern Matching
        complex_hits = 0
        for pattern in COMPLEX_PATTERNS:
            if pattern.search(full_text):
                complex_hits += 1

        if complex_hits >= 3:
            score += 0.40
            signals.append("dense_reasoning_or_code")
        elif complex_hits >= 1:
            score += 0.25
            signals.append("moderate_reasoning_signal")

        # 4. Simple Pattern Damping
        simple_hits = 0
        for pattern in SIMPLE_PATTERNS:
            if pattern.search(full_text):
                simple_hits += 1

        if simple_hits >= 1 and complex_hits == 0:
            score -= 0.25
            signals.append("simple_retrieval_or_summary")

        # Clamp score to [0.0, 1.0]
        final_score = max(0.0, min(1.0, round(score, 3)))
        return final_score, signals

