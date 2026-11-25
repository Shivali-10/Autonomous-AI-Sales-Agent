"""LLM integration (Gemini API primary, OpenAI optional, deterministic offline fallback)."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


load_dotenv()


class LLMService:
    """Unified wrapper for Gemini / OpenAI with graceful offline fallback."""

    def __init__(self) -> None:
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if self.gemini_api_key == "your_gemini_api_key_here":
            self.gemini_api_key = ""
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

        self._gemini_client = None
        self.last_error: str = ""
        self.provider_status: str = self._detect_provider()

    def _detect_provider(self) -> str:
        if self.gemini_api_key:
            try:
                from google import genai

                self._gemini_client = genai.Client(api_key=self.gemini_api_key)
                return f"Gemini API ({self.gemini_model})"
            except Exception as exc:
                self.last_error = str(exc)
        if self.openai_api_key:
            return f"OpenAI API ({self.openai_model})"
        return "Local Intelligent Fallback (No API Key)"

    @property
    def is_live_llm_available(self) -> bool:
        return bool(self._gemini_client or self.openai_api_key)

    def generate_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.4) -> Optional[str]:
        """Call configured LLM with error handling; returns None on failure so caller can fallback."""
        if self._gemini_client is not None:
            try:
                from google.genai import types

                response = self._gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                    ),
                )
                if response and getattr(response, "text", None):
                    self.last_error = ""
                    return response.text.strip()
            except Exception as exc:
                self.last_error = f"Gemini error: {exc}"

        if self.openai_api_key:
            try:
                import requests

                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.openai_model,
                        "temperature": temperature,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                    timeout=20,
                )
                resp.raise_for_status()
                data = resp.json()
                self.last_error = ""
                return data["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                self.last_error = f"OpenAI error: {exc}"

        return None

    def extract_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Generate structured JSON from the LLM with safe markdown fence stripping."""
        raw = self.generate_text(system_prompt, user_prompt, temperature=0.1)
        if not raw:
            return None
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    return None
        return None

    def transcribe_audio_gemini(self, audio_bytes: bytes, mime_type: str = "audio/wav") -> Optional[str]:
        """Optional Gemini multimodal audio transcription fallback."""
        if self._gemini_client is None or not audio_bytes:
            return None
        try:
            from google.genai import types

            response = self._gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                    "Transcribe the spoken audio accurately into plain text only. Do not add commentary.",
                ],
            )
            if response and getattr(response, "text", None):
                return response.text.strip()
        except Exception as exc:
            self.last_error = f"Gemini audio transcription error: {exc}"
        return None

    @staticmethod
    def build_fallback_sales_reply(
        user_message: str,
        lead_profile: Dict[str, Any],
        rag_chunks: List[Dict[str, Any]],
        sentiment_label: str,
        tone_guidance: str,
        booking_feedback: Optional[str] = None,
        language: str = "English",
    ) -> str:
        """Deterministic RAG-grounded conversational response when external LLM APIs are unavailable."""
        parts: List[str] = []
        name = lead_profile.get("name", "")
        greeting_name = name if name and name != "Unknown" else "there"

        if booking_feedback:
            parts.append(booking_feedback)

        if sentiment_label == "Negative":
            parts.append(
                f"I completely understand your concern, {greeting_name}, and I appreciate you sharing that openly."
            )
        elif sentiment_label == "Positive":
            parts.append(f"That sounds great, {greeting_name}!")

        lowered = user_message.lower()
        if any(g in lowered for g in ("hi", "hello", "hey", "namaste", "hola")) and len(user_message.split()) <= 6:
            parts.append(
                "Hello! I'm Aria, the AI Sales Representative for Fortivex AI. "
                "I can help you explore our Voice AI & CRM plans, estimate ROI, or schedule a live demo. "
                "Could you share your name and what business challenge you're looking to solve?"
            )
            return " ".join(parts)

        # Ground answer strictly in retrieved RAG chunks
        if rag_chunks and any(
            kw in lowered
            for kw in (
                "price", "cost", "plan", "starter", "growth", "enterprise",
                "feature", "language", "hindi", "crm", "security", "trial",
                "demo", "what", "how", "support", "minute", "add-on", "addon",
            )
        ):
            top = rag_chunks[0]["text"].strip()
            snippet = top[:460] + ("..." if len(top) > 460 else "")
            parts.append(f"Based on our official Fortivex AI documentation:\n{snippet}")
        elif not booking_feedback:
            parts.append(
                "Thank you for sharing those details. I've noted that in your profile so we can tailor the right Fortivex AI solution."
            )

        # Ask for next missing qualification field or offer appointment
        missing_prompts: List[str] = []
        if not lead_profile.get("name") or lead_profile.get("name") == "Unknown":
            missing_prompts.append("your name")
        if not lead_profile.get("requirements"):
            missing_prompts.append("your main use case or call volume")
        if not lead_profile.get("budget"):
            missing_prompts.append("your target monthly budget")
        if not lead_profile.get("timeline"):
            missing_prompts.append("your expected rollout timeline")
        if not lead_profile.get("email") and not lead_profile.get("phone") and not lead_profile.get("contact"):
            missing_prompts.append("your email or phone number")

        if missing_prompts and not booking_feedback:
            next_ask = ", ".join(missing_prompts[:2])
            parts.append(f"To recommend the best plan for you, could you please share {next_ask}?")
        elif not booking_feedback:
            parts.append(
                "You look like a strong fit for our platform! Would you like me to schedule a personalized demo "
                "(just share a preferred date and time, such as 'tomorrow at 3:00 PM')?"
            )

        if language and language != "English":
            parts.append(f"(Preferred language noted: {language}. Connect a Gemini API key in .env for full fluent {language} generation.)")

        return "\n\n".join(parts)
