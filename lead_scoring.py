"""Explainable 0-100 lead qualification and scoring engine."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class LeadScoreResult:
    score: int
    category: str  # Hot Lead | Warm Lead | Cold Lead
    breakdown: Dict[str, int]
    explanation: str
    missing_fields: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "category": self.category,
            "breakdown": self.breakdown,
            "explanation": self.explanation,
            "missing_fields": self.missing_fields,
        }


def parse_budget_amount(budget_text: str) -> float:
    """Extract an approximate monthly USD equivalent numeric value from budget text."""
    if not budget_text:
        return 0.0
    lowered = budget_text.lower().replace(",", "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*(k|lakh|lac|m)?", lowered)
    if not match:
        return 0.0
    val = float(match.group(1))
    suffix = match.group(2) or ""
    if suffix == "k":
        val *= 1000
    elif suffix in {"lakh", "lac"}:
        val *= 100000
    elif suffix == "m":
        val *= 1000000

    # Rough INR to USD conversion if INR / Rs / ₹ mentioned
    if any(curr in lowered for curr in ("inr", "rs", "₹", "rupee")):
        val = val / 83.0

    # Normalize annual budget to monthly if yearly indicated
    if any(period in lowered for period in ("year", "annual", "yr", "per annum")):
        val = val / 12.0

    return round(val, 2)


def _score_budget(budget: str, budget_amount: float) -> tuple[int, str]:
    lowered = (budget or "").lower()
    if not budget or any(w in lowered for w in ("not sure", "no budget", "exploring", "unknown")):
        return 3, "Budget not yet defined (3/25)."
    if budget_amount >= 1500:
        return 25, f"Enterprise-tier budget '{budget}' (25/25)."
    if budget_amount >= 500:
        return 22, f"Growth-tier budget '{budget}' (22/25)."
    if budget_amount >= 150:
        return 16, f"Starter-tier budget '{budget}' (16/25)."
    if budget_amount > 0:
        return 10, f"Limited budget indicated '{budget}' (10/25)."
    return 8, f"Budget mentioned '{budget}' without exact figure (8/25)."


def _score_timeline(timeline: str) -> tuple[int, str]:
    lowered = (timeline or "").lower()
    if not lowered:
        return 0, "Purchase timeline not shared yet (0/25)."
    if any(k in lowered for k in ("immediate", "asap", "this week", "today", "tomorrow", "this month", "1 week", "2 week", "now", "urgent")):
        return 25, f"Urgent timeline '{timeline}' (25/25)."
    if any(k in lowered for k in ("next month", "30 day", "1 month", "soon", "few weeks")):
        return 20, f"Near-term timeline '{timeline}' (20/25)."
    if any(k in lowered for k in ("quarter", "2 month", "3 month", "1-3 month", "q1", "q2", "q3", "q4")):
        return 14, f"Medium-term timeline '{timeline}' (14/25)."
    if any(k in lowered for k in ("research", "6 month", "next year", "later", "exploring", "not sure")):
        return 5, f"Long/exploratory timeline '{timeline}' (5/25)."
    return 12, f"Timeline noted as '{timeline}' (12/25)."


def _score_requirements(product_interest: str, requirements: str) -> tuple[int, str]:
    pts = 0
    if product_interest and product_interest.lower() not in ("unknown", "general"):
        pts += 8
    req_len = len((requirements or "").strip())
    if req_len >= 60:
        pts += 12
    elif req_len >= 20:
        pts += 8
    elif req_len > 0:
        pts += 4
    pts = min(20, pts)
    if pts >= 16:
        return pts, f"Clear product interest and specific requirements ({pts}/20)."
    if pts >= 8:
        return pts, f"Partial requirements captured ({pts}/20)."
    return pts, f"Requirements still need discovery ({pts}/20)."


def _score_contact(name: str, email: str, phone: str, contact: str) -> tuple[int, str]:
    pts = 0
    if name and name.lower() not in ("unknown", "guest", "customer"):
        pts += 5
    has_email = bool(email and "@" in email) or ("@" in (contact or ""))
    has_phone = bool(phone and len(re.sub(r"\D", "", phone)) >= 7) or bool(re.search(r"\d{7,}", contact or ""))
    if has_email:
        pts += 5
    if has_phone:
        pts += 5
    if pts == 15:
        return 15, "Complete identity & contact info (name, email, phone) (15/15)."
    if pts >= 10:
        return pts, f"Name and primary contact captured ({pts}/15)."
    if pts > 0:
        return pts, f"Partial contact info captured ({pts}/15)."
    return 0, "Contact info not yet provided (0/15)."


def _score_engagement(sentiment_label: str, turns_count: int, appointment_booked: bool) -> tuple[int, str]:
    pts = 0
    if sentiment_label == "Positive":
        pts += 8
    elif sentiment_label == "Neutral":
        pts += 5
    else:
        pts += 2

    if turns_count >= 3:
        pts += 4
    elif turns_count >= 1:
        pts += 2

    if appointment_booked:
        pts += 3

    pts = min(15, pts)
    return pts, f"Engagement & {sentiment_label.lower()} sentiment signal ({pts}/15)."


def score_lead(profile: Dict[str, Any], sentiment_label: str = "Neutral", turns_count: int = 1, appointment_booked: bool = False) -> LeadScoreResult:
    """Compute a transparent 0-100 lead score, category, and human-readable explanation."""
    budget = str(profile.get("budget", "") or "")
    budget_amount = float(profile.get("budget_amount", 0) or 0)
    if budget and not budget_amount:
        budget_amount = parse_budget_amount(budget)

    timeline = str(profile.get("timeline", "") or "")
    product_interest = str(profile.get("product_interest", "") or "")
    requirements = str(profile.get("requirements", "") or "")
    name = str(profile.get("name", "") or "")
    email = str(profile.get("email", "") or "")
    phone = str(profile.get("phone", "") or "")
    contact = str(profile.get("contact", "") or "")

    b_pts, b_msg = _score_budget(budget, budget_amount)
    t_pts, t_msg = _score_timeline(timeline)
    r_pts, r_msg = _score_requirements(product_interest, requirements)
    c_pts, c_msg = _score_contact(name, email, phone, contact)
    e_pts, e_msg = _score_engagement(sentiment_label, turns_count, appointment_booked)

    total = max(0, min(100, b_pts + t_pts + r_pts + c_pts + e_pts))

    if total >= 70:
        category = "Hot Lead"
    elif total >= 40:
        category = "Warm Lead"
    else:
        category = "Cold Lead"

    missing: List[str] = []
    if not name or name.lower() == "unknown":
        missing.append("Customer Name")
    if not email and not phone and not contact:
        missing.append("Contact Info (Email/Phone)")
    if not product_interest:
        missing.append("Product/Plan Interest")
    if not budget:
        missing.append("Budget")
    if not timeline:
        missing.append("Purchase Timeline")
    if not requirements:
        missing.append("Specific Requirements")

    breakdown = {
        "budget": b_pts,
        "timeline": t_pts,
        "requirements": r_pts,
        "contact": c_pts,
        "sentiment": e_pts,
    }

    explanation = f"Classified as {category} ({total}/100). " + " ".join([b_msg, t_msg, r_msg, c_msg, e_msg])
    if missing:
        explanation += f" Missing info to qualify further: {', '.join(missing)}."

    return LeadScoreResult(
        score=total,
        category=category,
        breakdown=breakdown,
        explanation=explanation,
        missing_fields=missing,
    )
