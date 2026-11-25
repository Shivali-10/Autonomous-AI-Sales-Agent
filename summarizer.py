"""End-of-call structured summarization module."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from database import save_call_summary
from llm_service import LLMService


def generate_call_summary(
    session_id: str,
    lead_id: Optional[int],
    lead_profile: Dict[str, Any],
    score_result: Dict[str, Any],
    messages: List[Dict[str, Any]],
    appointment_info: Optional[Dict[str, Any]],
    llm_service: Optional[LLMService] = None,
    db_path: Optional[str] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    """Build and optionally persist a complete post-call CRM summary."""
    customer_name = lead_profile.get("name") or "Unknown"
    requirements = lead_profile.get("requirements") or "Not explicitly specified during call."
    lead_score = int(score_result.get("score", 0))
    lead_category = score_result.get("category", "Cold Lead")

    # Compute overall customer sentiment from user turns
    user_sentiments = [m.get("sentiment", "Neutral") for m in messages if m.get("role") == "user"]
    if not user_sentiments:
        overall_sentiment = "Neutral"
    elif user_sentiments.count("Positive") >= user_sentiments.count("Negative") and user_sentiments.count("Positive") > 0:
        overall_sentiment = "Positive"
    elif user_sentiments.count("Negative") > user_sentiments.count("Positive"):
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    if appointment_info and appointment_info.get("confirmation_code"):
        appointment_status = (
            f"Confirmed (Simulated) on {appointment_info.get('appointment_date')} "
            f"at {appointment_info.get('appointment_time')} ({appointment_info.get('confirmation_code')})"
        )
    else:
        appointment_status = "Not Scheduled"

    # Try LLM summarization first if configured
    main_points = ""
    recommended_action = ""
    if llm_service and llm_service.is_live_llm_available and messages:
        transcript = "\n".join(
            f"{m.get('role', 'user').upper()}: {m.get('message', '')}" for m in messages[-16:]
        )
        prompt = (
            f"Lead Profile: {lead_profile}\n"
            f"Lead Score: {lead_score} ({lead_category})\n"
            f"Overall Sentiment: {overall_sentiment}\n"
            f"Appointment Status: {appointment_status}\n\n"
            f"Transcript:\n{transcript}\n\n"
            "Return JSON with keys: 'requirements', 'main_points', 'recommended_action'."
        )
        parsed = llm_service.extract_json(
            "You are a B2B SaaS CRM call summarizer. Return concise, accurate JSON only.",
            prompt,
        )
        if parsed:
            requirements = parsed.get("requirements") or requirements
            main_points = parsed.get("main_points") or ""
            recommended_action = parsed.get("recommended_action") or ""

    if not main_points:
        user_msgs = [m.get("message", "") for m in messages if m.get("role") == "user"]
        product = lead_profile.get("product_interest") or "Fortivex AI Voice Suite"
        budget = lead_profile.get("budget") or "budget TBD"
        timeline = lead_profile.get("timeline") or "timeline TBD"
        topics = "; ".join(msg[:90] for msg in user_msgs[:3]) if user_msgs else "Initial introduction"
        main_points = (
            f"Discussed {product} ({budget}, {timeline}). "
            f"Customer inquiries covered: {topics}."
        )

    if not recommended_action:
        if appointment_status != "Not Scheduled":
            recommended_action = (
                f"Send pre-demo agenda and architecture brief ahead of the scheduled session ({appointment_status})."
            )
        elif lead_category == "Hot Lead":
            recommended_action = "Immediate human AE follow-up within 2 hours to lock in a live product demo."
        elif lead_category == "Warm Lead":
            recommended_action = "Share tailored pricing comparison + 14-day sandbox trial link and follow up in 2 days."
        else:
            recommended_action = "Enroll in automated nurture email sequence with Fortivex AI ROI case studies."

    summary = {
        "session_id": session_id,
        "lead_id": lead_id,
        "customer_name": customer_name,
        "requirements": requirements,
        "main_points": main_points,
        "lead_score": lead_score,
        "lead_category": lead_category,
        "sentiment": overall_sentiment,
        "appointment_status": appointment_status,
        "recommended_action": recommended_action,
    }

    if persist:
        summary["id"] = save_call_summary(summary, db_path=db_path)

    return summary
