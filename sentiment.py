"""Customer sentiment analysis and conversational tone adaptation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


POSITIVE_WORDS = {
    "great", "good", "awesome", "excellent", "amazing", "love", "interested",
    "excited", "perfect", "helpful", "thanks", "thank", "wonderful", "yes",
    "sure", "definitely", "happy", "impressed", "affordable", "fast", "works",
    "book", "schedule", "proceed", "ready", "fantastic", "nice", "appreciate",
    "badhiya", "accha", "shukriya", "genial", "excelente",
}

NEGATIVE_WORDS = {
    "bad", "expensive", "costly", "overpriced", "slow", "confusing", "confused",
    "hate", "terrible", "worst", "no", "not", "never", "cancel", "problem",
    "issue", "frustrated", "angry", "disappointed", "complicated", "useless",
    "doubt", "unsure", "hard", "difficult", "annoying", "waste", "delay",
    "scam", "too high", "cannot", "can't", "won't", "nahi", "bekar",
}

STRONG_PHRASES = {
    "love it": 0.45,
    "sounds great": 0.40,
    "very interested": 0.45,
    "ready to buy": 0.50,
    "book a demo": 0.35,
    "schedule a call": 0.35,
    "too expensive": -0.50,
    "way too much": -0.45,
    "not interested": -0.55,
    "waste of time": -0.60,
    "doesn't work": -0.45,
    "no thanks": -0.35,
}

NEGATIONS = {"not", "no", "never", "neither", "don't", "doesn't", "didn't", "can't", "won't"}


@dataclass
class SentimentResult:
    label: str  # Positive | Neutral | Negative
    score: float  # -1.0 to +1.0
    confidence: float  # 0.0 to 1.0
    cues: List[str]
    tone_guidance: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "score": round(self.score, 2),
            "confidence": round(self.confidence, 2),
            "cues": self.cues,
            "tone_guidance": self.tone_guidance,
        }


def analyze_sentiment(text: str) -> SentimentResult:
    """Classify customer message sentiment and provide tone adaptation guidance."""
    if not text or not text.strip():
        return SentimentResult(
            label="Neutral",
            score=0.0,
            confidence=0.5,
            cues=[],
            tone_guidance="Keep a friendly, clear, and consultative sales tone.",
        )

    lowered = text.lower().strip()
    tokens = re.findall(r"[a-zA-Z']+", lowered)

    score = 0.0
    cues: List[str] = []

    for phrase, weight in STRONG_PHRASES.items():
        if phrase in lowered:
            score += weight
            cues.append(phrase)

    for idx, tok in enumerate(tokens):
        prev_negated = idx > 0 and tokens[idx - 1] in NEGATIONS
        if tok in POSITIVE_WORDS:
            delta = -0.25 if prev_negated else 0.25
            score += delta
            cues.append(f"{'not ' if prev_negated else ''}{tok}")
        elif tok in NEGATIVE_WORDS:
            if tok in NEGATIONS and idx + 1 < len(tokens) and tokens[idx + 1] in POSITIVE_WORDS:
                continue
            delta = 0.15 if prev_negated else -0.28
            score += delta
            cues.append(tok)

    if "!" in text and score > 0:
        score += 0.1

    score = max(-1.0, min(1.0, score))

    if score >= 0.18:
        label = "Positive"
        tone_guidance = (
            "Customer is positive and engaged. Match their enthusiasm, reinforce product value, "
            "and proactively guide toward next steps or booking a live demo."
        )
    elif score <= -0.18:
        label = "Negative"
        tone_guidance = (
            "Customer expresses concern, skepticism, or frustration. Acknowledge their concern with empathy, "
            "avoid pushiness, clarify ROI/pricing options transparently, and offer a low-commitment trial or Q&A."
        )
    else:
        label = "Neutral"
        tone_guidance = (
            "Customer is neutral or information-seeking. Maintain a helpful, consultative tone, "
            "answer concisely using verified knowledge, and ask one focused discovery question."
        )

    confidence = min(0.95, max(0.55, 0.55 + abs(score) * 0.4))
    unique_cues = list(dict.fromkeys(cues))[:5]

    return SentimentResult(
        label=label,
        score=score,
        confidence=confidence,
        cues=unique_cues,
        tone_guidance=tone_guidance,
    )
