"""Speech-to-Text (STT) and Text-to-Speech (TTS) service with safe Windows & offline fallbacks."""

from __future__ import annotations

import io
from typing import Dict, Optional, Tuple

import speech_recognition as sr
from gtts import gTTS

from llm_service import LLMService


LANGUAGE_CODES: Dict[str, Tuple[str, str]] = {
    "English": ("en-IN", "en"),
    "Hindi": ("hi-IN", "hi"),
    "Hinglish": ("en-IN", "en"),
    "Spanish": ("es-ES", "es"),
    "French": ("fr-FR", "fr"),
    "German": ("de-DE", "de"),
    "Tamil": ("ta-IN", "ta"),
    "Telugu": ("te-IN", "te"),
}


def transcribe_audio_bytes(
    audio_bytes: bytes,
    language: str = "English",
    llm_service: Optional[LLMService] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert WAV audio bytes (from Streamlit audio_input or file upload) into text.
    Returns (transcribed_text, error_message).
    """
    if not audio_bytes:
        return None, "No audio data received."

    stt_lang, _ = LANGUAGE_CODES.get(language, ("en-IN", "en"))

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language=stt_lang)
        if text and text.strip():
            return text.strip(), None
    except sr.UnknownValueError:
        # Try Gemini multimodal fallback if available
        if llm_service and llm_service.is_live_llm_available:
            gemini_text = llm_service.transcribe_audio_gemini(audio_bytes)
            if gemini_text:
                return gemini_text, None
        return None, "Could not understand the speech clearly. Please try speaking closer to the mic or use text chat."
    except sr.RequestError as exc:
        if llm_service and llm_service.is_live_llm_available:
            gemini_text = llm_service.transcribe_audio_gemini(audio_bytes)
            if gemini_text:
                return gemini_text, None
        return None, f"Speech recognition service unavailable ({exc}). You can use text chat fallback."
    except Exception as exc:
        if llm_service and llm_service.is_live_llm_available:
            gemini_text = llm_service.transcribe_audio_gemini(audio_bytes)
            if gemini_text:
                return gemini_text, None
        return None, f"Audio format error ({exc}). Please record via the microphone widget or upload a valid WAV file."

    return None, "No speech detected in audio."


def synthesize_speech_bytes(text: str, language: str = "English") -> Tuple[Optional[bytes], Optional[str]]:
    """
    Convert response text into MP3 audio bytes using gTTS (with optional pyttsx3 WAV fallback).
    Returns (audio_bytes, error_message).
    """
    if not text or not text.strip():
        return None, "Empty text cannot be synthesized."

    # Keep spoken output concise and clean of markdown formatting
    clean_text = (
        text.replace("**", "")
        .replace("##", "")
        .replace("#", "")
        .replace("`", "")
        .replace("...", ".")
    )
    if len(clean_text) > 900:
        clean_text = clean_text[:900] + "."

    _, tts_lang = LANGUAGE_CODES.get(language, ("en-IN", "en"))

    try:
        buf = io.BytesIO()
        tts = gTTS(text=clean_text, lang=tts_lang, slow=False)
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read(), None
    except Exception as gtts_exc:
        # Optional offline fallback with pyttsx3 if installed on Windows
        try:
            import os
            import tempfile
            import pyttsx3

            engine = pyttsx3.init()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            engine.save_to_file(clean_text, tmp_path)
            engine.runAndWait()
            with open(tmp_path, "rb") as f:
                wav_bytes = f.read()
            os.remove(tmp_path)
            return wav_bytes, None
        except Exception:
            return None, f"TTS unavailable offline ({gtts_exc}). Text response is displayed below."
