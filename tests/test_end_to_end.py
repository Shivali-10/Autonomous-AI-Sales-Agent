"""End-to-end verification tests for the Autonomous AI Sales Voice Agent MVP."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from agent import SalesVoiceAgent
from appointment import book_appointment, parse_date_time
from database import get_all_leads, get_appointments, get_call_summaries, get_dashboard_metrics, init_db
from lead_scoring import score_lead
from llm_service import LLMService
from rag import KnowledgeBaseRAG
from sentiment import analyze_sentiment
from summarizer import generate_call_summary


def test_full_end_to_end_workflow(tmp_path: Path) -> None:
    db_file = str(tmp_path / "test_sales.db")
    index_dir = str(tmp_path / "faiss_idx")
    kb_dir = str(ROOT_DIR / "knowledge_base")

    # 1. Initialize database cleanly without sample seed so assertions are exact
    init_db(db_path=db_file, seed_sample=False)

    # 2. Initialize RAG, LLM (offline fallback safe), and Agent
    rag = KnowledgeBaseRAG(kb_dir=kb_dir, index_dir=index_dir, db_path=db_file)
    llm = LLMService()
    agent = SalesVoiceAgent(rag=rag, llm=llm, db_path=db_file)

    assert len(rag.chunks) > 0, "Knowledge base should produce indexed chunks"
    rag_hits = rag.search("Growth plan price and Hindi support", top_k=3)
    assert len(rag_hits) > 0, "RAG search should return relevant chunks"

    session_id = "test-session-e2e"
    profile = SalesVoiceAgent.default_profile(session_id)
    history = []

    # Turn 1: Greeting + Product Inquiry
    t1 = agent.process_turn(
        session_id=session_id,
        user_message="Hi, I'm Ananya from Zenith Health. We need an English and Hindi AI voice agent for 3,000 calls per month. What plans do you offer?",
        lead_profile=profile,
        history=history,
    )
    profile = t1["lead_profile"]
    history.append({"role": "user", "message": "turn 1", "sentiment": t1["sentiment"]["label"]})
    history.append({"role": "assistant", "message": t1["reply"]})
    assert profile["name"] == "Ananya"
    assert len(t1["rag_sources"]) > 0

    # Turn 2: Budget + Timeline + Contact
    t2 = agent.process_turn(
        session_id=session_id,
        user_message="Growth Pro sounds great! Our budget is $600/month, timeline is this month, email is ananya@zenithhealth.in and phone 9876501234.",
        lead_profile=profile,
        history=history,
    )
    profile = t2["lead_profile"]
    history.append({"role": "user", "message": "turn 2", "sentiment": t2["sentiment"]["label"]})
    history.append({"role": "assistant", "message": t2["reply"]})
    assert t2["lead_score"]["score"] >= 70
    assert t2["lead_score"]["category"] == "Hot Lead"
    assert t2["sentiment"]["label"] == "Positive"

    # Turn 3: Appointment Request (ensure future weekday Mon-Sat)
    next_day = datetime.now() + timedelta(days=1)
    if next_day.weekday() == 6:  # Sunday -> Monday
        next_day += timedelta(days=1)
    target_date = next_day.strftime("%Y-%m-%d")

    t3 = agent.process_turn(
        session_id=session_id,
        user_message=f"Please book a demo for {target_date} at 3:00 PM.",
        lead_profile=profile,
        history=history,
    )
    profile = t3["lead_profile"]
    appt = t3["appointment"]
    assert appt is not None and appt["confirmation_code"].startswith("FTX-")

    # Conflict check on same slot
    conflict = book_appointment(
        session_id="conflict-test",
        lead_id=None,
        customer_name="Other User",
        customer_contact="other@example.com",
        appointment_date=target_date,
        appointment_time="15:00",
        db_path=db_file,
    )
    assert conflict.success is False

    # 4. Generate & persist Call Summary
    summary = generate_call_summary(
        session_id=session_id,
        lead_id=profile["id"],
        lead_profile=profile,
        score_result=t3["lead_score"],
        messages=history,
        appointment_info=appt,
        llm_service=llm,
        db_path=db_file,
        persist=True,
    )
    assert summary["customer_name"] == "Ananya"
    assert "FTX-" in summary["appointment_status"]

    # 5. Verify SQLite CRM Dashboard State
    assert len(get_all_leads(db_path=db_file)) == 1
    assert len(get_appointments(db_path=db_file)) == 1
    assert len(get_call_summaries(db_path=db_file)) == 1
    metrics = get_dashboard_metrics(db_path=db_file)
    assert metrics["hot_leads"] == 1
    assert metrics["total_appointments"] == 1


def test_sentiment_and_scoring_edge_cases() -> None:
    neg = analyze_sentiment("This is way too expensive and confusing, not interested.")
    assert neg.label == "Negative"
    assert neg.score < 0

    cold = score_lead({"name": "Unknown"}, sentiment_label="Negative", turns_count=1)
    assert cold.category == "Cold Lead"
    assert cold.score < 40

    d_str, t_str = parse_date_time(None, None, raw_text="Can we meet tomorrow at 4:30 pm?")
    assert d_str is not None
    assert t_str == "16:30"
