"""Autonomous AI Sales Agent orchestrator with contextual memory, RAG, scoring, and booking."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from appointment import book_appointment, parse_date_time
from database import add_conversation_turn, upsert_lead
from lead_scoring import parse_budget_amount, score_lead
from llm_service import LLMService
from rag import KnowledgeBaseRAG
from sentiment import analyze_sentiment


class SalesVoiceAgent:
    """Coordinates every step of the customer sales conversation and CRM synchronization."""

    def __init__(
        self,
        rag: KnowledgeBaseRAG,
        llm: LLMService,
        db_path: Optional[str] = None,
    ) -> None:
        self.rag = rag
        self.llm = llm
        self.db_path = db_path

    @staticmethod
    def default_profile(session_id: str, language: str = "English") -> Dict[str, Any]:
        return {
            "session_id": session_id,
            "name": "Unknown",
            "email": "",
            "phone": "",
            "contact": "",
            "company": "",
            "product_interest": "",
            "budget": "",
            "budget_amount": 0.0,
            "timeline": "",
            "requirements": "",
            "preferred_language": language,
            "status": "Active",
        }

    def _extract_entities_regex(self, text: str) -> Dict[str, Any]:
        """Deterministic entity extraction fallback that works reliably offline."""
        extracted: Dict[str, Any] = {}

        # Email
        email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
        if email_match:
            extracted["email"] = email_match.group(0)

        # Phone
        phone_match = re.search(r"(?:\+?\d{1,3}[\s-]?)?\b\d{10}\b", text)
        if phone_match:
            extracted["phone"] = phone_match.group(0)

        # Name patterns (keep name group case-sensitive so lowercase words like 'from' don't match)
        name_patterns = [
            r"(?i:my name is|i am|i'm|this is|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"(?i:name\s*[:\-]\s*)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        ]
        stop_words = {
            "interested", "looking", "from", "calling", "here", "ready",
            "fine", "good", "sorry", "sure", "booking", "scheduling", "at", "with",
        }
        for pat in name_patterns:
            m = re.search(pat, text)
            if m:
                tokens = [tok for tok in m.group(1).strip().split() if tok.lower() not in stop_words]
                if tokens:
                    extracted["name"] = " ".join(tokens).title()
                break

        # Company
        comp_match = re.search(
            r"(?:from|at|company\s*(?:is|:))\s+([A-Z][A-Za-z0-9&.\- ]{2,35})",
            text,
        )
        if comp_match:
            extracted["company"] = comp_match.group(1).strip(" .,")

        # Budget
        budget_match = re.search(
            r"(?:budget(?:\s+is|\s+around|\s+of)?\s*(?:is|:)?\s*|spend\s+|around\s+)?((?:\$|₹|rs\.?|inr|usd)\s*\d[\d,]*(?:\.\d+)?\s*(?:k|lakh|lac)?(?:\s*/\s*(?:mo|month|yr|year)|(?:\s+per\s+(?:month|year)))?)",
            text,
            flags=re.IGNORECASE,
        )
        if budget_match:
            extracted["budget"] = budget_match.group(1).strip()

        # Timeline
        timeline_patterns = [
            r"\b(immediate(?:ly)?|asap|this week|next week|this month|next month|within \d+ (?:days?|weeks?|months?)|next quarter|1-3 months|6\+?\s*months|next year|just researching|tomorrow|today)\b"
        ]
        for pat in timeline_patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                extracted["timeline"] = m.group(1).strip().title()
                break

        # Product interest
        lowered = text.lower()
        if "enterprise" in lowered:
            extracted["product_interest"] = "Fortivex Enterprise Autonomous Platform"
        elif "growth" in lowered or "crm" in lowered or "pro" in lowered:
            extracted["product_interest"] = "Fortivex Growth AI CRM & Voice Pro"
        elif "starter" in lowered or "clinic" in lowered or "small" in lowered:
            extracted["product_interest"] = "Fortivex Starter Voice Agent"
        elif "voice" in lowered or "agent" in lowered or "sales" in lowered:
            extracted["product_interest"] = "Fortivex AI Voice Agent"

        # Appointment intent + date/time
        booking_keywords = ("book", "schedule", "appointment", "demo", "meeting", "call slot", "slot")
        if any(k in lowered for k in booking_keywords):
            extracted["wants_appointment"] = True
        d_str, t_str = parse_date_time(None, None, raw_text=text)
        if d_str:
            extracted["preferred_date"] = d_str
        if t_str:
            extracted["preferred_time"] = t_str

        return extracted

    def _extract_entities(self, user_message: str, current_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Combine regex extraction with LLM JSON extraction for high accuracy."""
        merged = self._extract_entities_regex(user_message)

        if self.llm.is_live_llm_available:
            sys_prompt = (
                "Extract sales lead entities from the user's latest message given the current profile. "
                "Return ONLY valid JSON with keys: name, email, phone, company, product_interest, "
                "budget, timeline, requirements, wants_appointment (bool), preferred_date (YYYY-MM-DD or ''), "
                "preferred_time (HH:MM 24h or ''), intent (short label). Use empty string '' if not mentioned."
            )
            usr_prompt = f"Current Profile: {current_profile}\nLatest User Message: {user_message}"
            llm_data = self.llm.extract_json(sys_prompt, usr_prompt)
            if isinstance(llm_data, dict):
                for key, val in llm_data.items():
                    if val and str(val).strip() and str(val).strip().lower() not in {"unknown", "none", "null", "n/a"}:
                        if key not in merged or not merged[key]:
                            merged[key] = val

        return merged

    @staticmethod
    def _merge_profile(profile: Dict[str, Any], extracted: Dict[str, Any], user_message: str) -> Dict[str, Any]:
        """Update contextual memory without erasing previously learned fields."""
        updated = dict(profile)
        for field in ("name", "email", "phone", "company", "product_interest", "budget", "timeline"):
            val = extracted.get(field)
            if val and str(val).strip():
                updated[field] = str(val).strip()

        # Compose unified contact string
        contacts = [c for c in [updated.get("email", ""), updated.get("phone", "")] if c]
        if contacts:
            updated["contact"] = " | ".join(contacts)

        if updated.get("budget"):
            updated["budget_amount"] = parse_budget_amount(updated["budget"])

        # Append substantive requirement statements
        req_text = str(extracted.get("requirements", "") or "").strip()
        if not req_text and len(user_message.split()) >= 5 and any(
            kw in user_message.lower()
            for kw in ("need", "want", "looking for", "require", "calls", "crm", "language", "support", "agent", "automate", "team")
        ):
            req_text = user_message.strip()

        if req_text:
            existing_req = updated.get("requirements", "").strip()
            if req_text not in existing_req:
                updated["requirements"] = f"{existing_req}; {req_text}".strip("; ") if existing_req else req_text

        return updated

    def process_turn(
        self,
        session_id: str,
        user_message: str,
        lead_profile: Dict[str, Any],
        history: List[Dict[str, Any]],
        language: str = "English",
        existing_appointment: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a complete end-to-end conversational turn and update CRM state."""
        lead_profile["preferred_language"] = language

        # 1. Analyze sentiment
        sentiment_res = analyze_sentiment(user_message)

        # 2. Extract entities & update contextual memory
        extracted = self._extract_entities(user_message, lead_profile)
        updated_profile = self._merge_profile(lead_profile, extracted, user_message)

        # 3. Retrieve grounded company knowledge via FAISS RAG
        rag_results = self.rag.search(user_message, top_k=4)
        rag_dicts = [r.to_dict() for r in rag_results]
        rag_sources = list(dict.fromkeys(r.source for r in rag_results))

        # 4. Score lead (preliminary before checking if booking succeeds)
        turns_count = sum(1 for m in history if m.get("role") == "user") + 1
        appt_booked = existing_appointment is not None
        score_res = score_lead(
            updated_profile,
            sentiment_label=sentiment_res.label,
            turns_count=turns_count,
            appointment_booked=appt_booked,
        )

        # Persist/update lead first to obtain lead_id
        updated_profile.update(
            {
                "lead_score": score_res.score,
                "category": score_res.category,
                "score_breakdown": score_res.breakdown,
                "score_explanation": score_res.explanation,
            }
        )
        lead_id = upsert_lead(updated_profile, db_path=self.db_path)
        updated_profile["id"] = lead_id

        # 5. Check appointment booking workflow
        booking_feedback: Optional[str] = None
        new_appointment: Optional[Dict[str, Any]] = existing_appointment
        pref_date = extracted.get("preferred_date", "")
        pref_time = extracted.get("preferred_time", "")
        wants_booking = bool(extracted.get("wants_appointment")) or bool(pref_date and pref_time)

        if wants_booking and not existing_appointment:
            if pref_date and pref_time:
                booking_res = book_appointment(
                    session_id=session_id,
                    lead_id=lead_id,
                    customer_name=updated_profile.get("name", ""),
                    customer_contact=updated_profile.get("contact", ""),
                    appointment_date=pref_date,
                    appointment_time=pref_time,
                    purpose=f"Demo: {updated_profile.get('product_interest') or 'Fortivex AI Voice Suite'}",
                    notes=updated_profile.get("requirements", ""),
                    db_path=self.db_path,
                )
                booking_feedback = booking_res.message
                if booking_res.success:
                    new_appointment = booking_res.to_dict()
                    updated_profile["status"] = "Demo Scheduled"
                    score_res = score_lead(
                        updated_profile,
                        sentiment_label=sentiment_res.label,
                        turns_count=turns_count,
                        appointment_booked=True,
                    )
                    updated_profile.update(
                        {
                            "lead_score": score_res.score,
                            "category": score_res.category,
                            "score_breakdown": score_res.breakdown,
                            "score_explanation": score_res.explanation,
                        }
                    )
                    upsert_lead(updated_profile, db_path=self.db_path)
            else:
                booking_feedback = (
                    "I'd be happy to schedule your demo! Please share your preferred date and time "
                    "(Monday-Saturday, 9:00 AM to 7:00 PM) along with your name and email/phone."
                )

        # 6. Persist user turn
        intent = str(extracted.get("intent") or ("appointment_request" if wants_booking else "sales_inquiry"))
        add_conversation_turn(
            session_id=session_id,
            lead_id=lead_id,
            role="user",
            message=user_message,
            sentiment=sentiment_res.label,
            sentiment_score=sentiment_res.score,
            rag_sources=rag_sources,
            intent=intent,
            db_path=self.db_path,
        )

        # 7. Generate grounded, sentiment-aware sales response
        rag_context = "\n\n".join(
            f"[Source: {r.source} | Score: {r.score:.2f}]\n{r.text}" for r in rag_results
        ) or "No knowledge documents retrieved."

        recent_history = "\n".join(
            f"{m.get('role', 'user').title()}: {m.get('message', '')}" for m in history[-8:]
        )

        system_prompt = (
            "You are Aria, an autonomous, empathetic, and sharp AI Sales Representative for Fortivex AI.\n"
            f"Respond in {language}.\n"
            "STRICT RULES:\n"
            "1. Only state product features, prices, and policies found in the Retrieved Knowledge Context below. Never invent prices or features.\n"
            "2. Use the Customer Profile & Contextual Memory to avoid re-asking known details.\n"
            f"3. Sentiment Adaptation ({sentiment_res.label}): {sentiment_res.tone_guidance}\n"
            "4. Keep responses concise (3-5 sentences), natural for voice playback, and ask at most one clear follow-up question.\n"
            "5. If a simulated appointment was booked or needs missing fields, communicate the exact booking feedback clearly and honestly.\n"
        )

        user_prompt = (
            f"Customer Profile Memory: {updated_profile}\n"
            f"Lead Score: {score_res.score}/100 ({score_res.category})\n"
            f"Missing Qualification Info: {score_res.missing_fields}\n"
            f"Booking Status Feedback: {booking_feedback or 'None'}\n\n"
            f"Retrieved Knowledge Context:\n{rag_context}\n\n"
            f"Recent Conversation:\n{recent_history}\n"
            f"Customer: {user_message}\n\n"
            "Reply as Aria:"
        )

        reply = self.llm.generate_text(system_prompt, user_prompt, temperature=0.35)
        used_fallback = False
        if not reply:
            used_fallback = True
            reply = self.llm.build_fallback_sales_reply(
                user_message=user_message,
                lead_profile=updated_profile,
                rag_chunks=rag_dicts,
                sentiment_label=sentiment_res.label,
                tone_guidance=sentiment_res.tone_guidance,
                booking_feedback=booking_feedback,
                language=language,
            )

        # 8. Persist assistant turn
        add_conversation_turn(
            session_id=session_id,
            lead_id=lead_id,
            role="assistant",
            message=reply,
            sentiment="Neutral",
            sentiment_score=0.0,
            rag_sources=rag_sources,
            intent="agent_response",
            db_path=self.db_path,
        )

        return {
            "reply": reply,
            "used_fallback": used_fallback,
            "sentiment": sentiment_res.to_dict(),
            "lead_profile": updated_profile,
            "lead_score": score_res.to_dict(),
            "rag_chunks": rag_dicts,
            "rag_sources": rag_sources,
            "appointment": new_appointment,
            "booking_feedback": booking_feedback,
            "intent": intent,
        }
