"""SQLite persistence layer for leads, conversations, appointments, summaries, and knowledge docs."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional


DEFAULT_DB_PATH = os.getenv("DATABASE_PATH", "data/sales_agent.db")


def get_db_path(db_path: Optional[str] = None) -> Path:
    """Resolve and ensure the parent directory exists for the SQLite database."""
    resolved = Path(db_path or os.getenv("DATABASE_PATH", DEFAULT_DB_PATH))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


@contextmanager
def get_connection(db_path: Optional[str] = None) -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection configured with row factory and foreign keys."""
    path = get_db_path(db_path)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[str] = None, seed_sample: Optional[bool] = None) -> None:
    """Initialize SQLite schema automatically and optionally seed sample demo data."""
    if seed_sample is None:
        seed_sample = os.getenv("ENABLE_SAMPLE_SEED", "true").lower() == "true"

    with get_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                name TEXT DEFAULT 'Unknown',
                contact TEXT DEFAULT '',
                email TEXT DEFAULT '',
                phone TEXT DEFAULT '',
                company TEXT DEFAULT '',
                product_interest TEXT DEFAULT '',
                budget TEXT DEFAULT '',
                budget_amount REAL DEFAULT 0,
                timeline TEXT DEFAULT '',
                requirements TEXT DEFAULT '',
                preferred_language TEXT DEFAULT 'English',
                lead_score INTEGER DEFAULT 0,
                category TEXT DEFAULT 'Cold Lead',
                score_breakdown TEXT DEFAULT '{}',
                score_explanation TEXT DEFAULT '',
                status TEXT DEFAULT 'New',
                is_sample INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                lead_id INTEGER,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                sentiment TEXT DEFAULT 'Neutral',
                sentiment_score REAL DEFAULT 0.0,
                rag_sources TEXT DEFAULT '[]',
                intent TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (lead_id) REFERENCES leads(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                lead_id INTEGER,
                customer_name TEXT NOT NULL,
                customer_contact TEXT DEFAULT '',
                appointment_date TEXT NOT NULL,
                appointment_time TEXT NOT NULL,
                purpose TEXT DEFAULT 'Product Demo & Consultation',
                notes TEXT DEFAULT '',
                status TEXT DEFAULT 'Confirmed (Simulated)',
                confirmation_code TEXT UNIQUE NOT NULL,
                is_simulated INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                FOREIGN KEY (lead_id) REFERENCES leads(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS call_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                lead_id INTEGER,
                customer_name TEXT NOT NULL,
                requirements TEXT DEFAULT '',
                main_points TEXT DEFAULT '',
                lead_score INTEGER DEFAULT 0,
                lead_category TEXT DEFAULT 'Cold Lead',
                sentiment TEXT DEFAULT 'Neutral',
                appointment_status TEXT DEFAULT 'Not Scheduled',
                recommended_action TEXT DEFAULT '',
                full_summary_json TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (lead_id) REFERENCES leads(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                char_count INTEGER DEFAULT 0,
                uploaded_at TEXT NOT NULL
            );
            """
        )

    if seed_sample:
        seed_sample_data(db_path=db_path)


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def seed_sample_data(db_path: Optional[str] = None, force: bool = False) -> bool:
    """Populate realistic sample CRM leads, conversations, appointments, and call summaries if empty."""
    with get_connection(db_path) as conn:
        existing_count = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
        if existing_count > 0 and not force:
            return False

        if force:
            conn.executescript(
                """
                DELETE FROM call_summaries;
                DELETE FROM appointments;
                DELETE FROM conversations;
                DELETE FROM leads;
                """
            )

        base_dt = datetime.now()
        d1 = (base_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        d2 = (base_dt + timedelta(days=2)).strftime("%Y-%m-%d")
        d3 = (base_dt + timedelta(days=4)).strftime("%Y-%m-%d")

        sample_leads = [
            {
                "session_id": "sample-session-001",
                "name": "Aarav Mehta",
                "contact": "aarav.mehta@finbridge.in | +91-9876543210",
                "email": "aarav.mehta@finbridge.in",
                "phone": "+91-9876543210",
                "company": "FinBridge Capital",
                "product_interest": "Fortivex Growth AI CRM & Voice Pro",
                "budget": "$600/month ($7,200/year)",
                "budget_amount": 600.0,
                "timeline": "Immediate (This Month)",
                "requirements": "Multilingual inbound sales voice agent (English + Hindi) with Zoho CRM sync and appointment scheduling for 4,000 calls/month.",
                "preferred_language": "English",
                "lead_score": 88,
                "category": "Hot Lead",
                "score_breakdown": {
                    "budget": 23,
                    "timeline": 25,
                    "requirements": 18,
                    "contact": 15,
                    "sentiment": 7,
                },
                "score_explanation": "Strong monthly budget ($600/mo aligns with Growth Pro), immediate purchase timeline, clear multi-language CRM requirements, complete contact details, and positive engagement.",
                "status": "Demo Scheduled",
                "created_at": (base_dt - timedelta(hours=18)).strftime("%Y-%m-%d %H:%M:%S"),
            },
            {
                "session_id": "sample-session-002",
                "name": "Priya Nair",
                "contact": "priya@medcareclinics.com",
                "email": "priya@medcareclinics.com",
                "phone": "",
                "company": "MedCare Clinics",
                "product_interest": "Fortivex Starter Voice Agent",
                "budget": "$200/month",
                "budget_amount": 200.0,
                "timeline": "Next Quarter (1-3 Months)",
                "requirements": "Automated patient appointment booking and FAQ voice assistant in English and Tamil.",
                "preferred_language": "English",
                "lead_score": 64,
                "category": "Warm Lead",
                "score_breakdown": {
                    "budget": 16,
                    "timeline": 15,
                    "requirements": 16,
                    "contact": 10,
                    "sentiment": 7,
                },
                "score_explanation": "Clear budget fit for Starter plan ($199/mo) and defined clinic scheduling requirements, though timeline is 1-3 months and phone number was not shared yet.",
                "status": "Qualified",
                "created_at": (base_dt - timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S"),
            },
            {
                "session_id": "sample-session-003",
                "name": "Rohan Verma",
                "contact": "rohan.v@edunova.org | +91-9123456789",
                "email": "rohan.v@edunova.org",
                "phone": "+91-9123456789",
                "company": "EduNova Academy",
                "product_interest": "Fortivex Enterprise Autonomous Platform",
                "budget": "$2,200/month",
                "budget_amount": 2200.0,
                "timeline": "Within 2 Weeks",
                "requirements": "High-volume outbound admission counseling voice agents with RAG on course catalog and Salesforce integration.",
                "preferred_language": "English",
                "lead_score": 94,
                "category": "Hot Lead",
                "score_breakdown": {
                    "budget": 25,
                    "timeline": 25,
                    "requirements": 20,
                    "contact": 15,
                    "sentiment": 9,
                },
                "score_explanation": "Enterprise-level budget ($2,200/mo), urgent 2-week rollout target, complete email + phone contact, detailed RAG/outbound requirements, and enthusiastic sentiment.",
                "status": "Proposal Sent",
                "created_at": (base_dt - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
            },
            {
                "session_id": "sample-session-004",
                "name": "Vikram Kulkarni",
                "contact": "vikram@craftretail.in",
                "email": "vikram@craftretail.in",
                "phone": "",
                "company": "CraftRetail",
                "product_interest": "General Exploration",
                "budget": "Exploring / Not Decided",
                "budget_amount": 0.0,
                "timeline": "6+ Months / Just Researching",
                "requirements": "Wanted to understand how AI voice agents work and compare pricing against manual call centers.",
                "preferred_language": "English",
                "lead_score": 31,
                "category": "Cold Lead",
                "score_breakdown": {
                    "budget": 4,
                    "timeline": 5,
                    "requirements": 8,
                    "contact": 9,
                    "sentiment": 5,
                },
                "score_explanation": "Early research phase with no confirmed budget or near-term timeline; email captured for nurture campaign.",
                "status": "Nurture",
                "created_at": (base_dt - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S"),
            },
        ]

        lead_ids: Dict[str, int] = {}
        for item in sample_leads:
            cur = conn.execute(
                """
                INSERT INTO leads (
                    session_id, name, contact, email, phone, company,
                    product_interest, budget, budget_amount, timeline, requirements,
                    preferred_language, lead_score, category, score_breakdown,
                    score_explanation, status, is_sample, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    item["session_id"],
                    item["name"],
                    item["contact"],
                    item["email"],
                    item["phone"],
                    item["company"],
                    item["product_interest"],
                    item["budget"],
                    item["budget_amount"],
                    item["timeline"],
                    item["requirements"],
                    item["preferred_language"],
                    item["lead_score"],
                    item["category"],
                    json.dumps(item["score_breakdown"]),
                    item["score_explanation"],
                    item["status"],
                    item["created_at"],
                    item["created_at"],
                ),
            )
            lead_ids[item["session_id"]] = int(cur.lastrowid)

        sample_messages = [
            ("sample-session-001", "user", "Hi, I'm Aarav from FinBridge Capital. We need an AI voice agent supporting Hindi and English with Zoho CRM sync.", "Positive", 0.65, "product_inquiry"),
            ("sample-session-001", "assistant", "Welcome Aarav! Our Growth AI CRM & Voice Pro plan ($599/month) supports 15+ languages including Hindi and English with native Zoho CRM sync. What monthly call volume and timeline are you planning?", "Neutral", 0.2, "qualification"),
            ("sample-session-001", "user", "Around 4,000 calls per month, budget is $600/month, and we want to launch this month. Can we book a demo tomorrow at 11:00 AM?", "Positive", 0.78, "appointment_request"),
            ("sample-session-002", "user", "Hello, I'm Priya from MedCare Clinics. We are looking for an affordable appointment booking voice agent around $200/month for next quarter.", "Neutral", 0.15, "product_inquiry"),
            ("sample-session-003", "user", "Hi! Rohan from EduNova Academy here. We need an Enterprise voice AI for outbound student admissions within 2 weeks, budget around $2,200/month.", "Positive", 0.85, "high_intent_inquiry"),
            ("sample-session-004", "user", "Just browsing to see what voice AI costs compared to human callers. Not sure when we'd buy.", "Negative", -0.25, "price_objection"),
        ]
        for sid, role, msg, sent, s_score, intent in sample_messages:
            conn.execute(
                """
                INSERT INTO conversations (session_id, lead_id, role, message, sentiment, sentiment_score, rag_sources, intent, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (sid, lead_ids[sid], role, msg, sent, s_score, json.dumps(["fortivex_product_catalog.md"]), intent, _now_iso()),
            )

        sample_appointments = [
            ("sample-session-001", lead_ids["sample-session-001"], "Aarav Mehta", "aarav.mehta@finbridge.in", d1, "11:00", "Growth Plan Live Demo & Zoho Sync Walkthrough", "Sample simulated booking", "Confirmed (Simulated)", "FTX-DEMO-1001"),
            ("sample-session-003", lead_ids["sample-session-003"], "Rohan Verma", "rohan.v@edunova.org", d2, "15:30", "Enterprise Architecture & Outbound Dialer Review", "Sample simulated booking", "Confirmed (Simulated)", "FTX-DEMO-1002"),
            ("sample-session-002", lead_ids["sample-session-002"], "Priya Nair", "priya@medcareclinics.com", d3, "10:00", "Starter Plan Clinic Scheduling Demo", "Sample simulated booking", "Pending (Simulated)", "FTX-DEMO-1003"),
        ]
        for sid, lid, cname, ccontact, adate, atime, purpose, notes, status, code in sample_appointments:
            conn.execute(
                """
                INSERT INTO appointments (
                    session_id, lead_id, customer_name, customer_contact,
                    appointment_date, appointment_time, purpose, notes,
                    status, confirmation_code, is_simulated, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (sid, lid, cname, ccontact, adate, atime, purpose, notes, status, code, _now_iso()),
            )

        sample_summaries = [
            (
                "sample-session-001",
                lead_ids["sample-session-001"],
                "Aarav Mehta",
                "Hindi + English inbound sales voice agent for 4,000 calls/month with Zoho CRM integration.",
                "Discussed Growth AI CRM & Voice Pro ($599/month), confirmed budget fit ($600/month), and scheduled a demo.",
                88,
                "Hot Lead",
                "Positive",
                f"Confirmed (Simulated) for {d1} at 11:00 (FTX-DEMO-1001)",
                "Prepare Zoho CRM sandbox demo and multilingual Hindi/English call flow before tomorrow's call.",
            ),
            (
                "sample-session-003",
                lead_ids["sample-session-003"],
                "Rohan Verma",
                "High-volume outbound admission voice agents with RAG on course catalog and Salesforce integration.",
                "Reviewed Enterprise Autonomous Platform ($1,999+/month), confirmed $2,200/month budget and 2-week rollout timeline.",
                94,
                "Hot Lead",
                "Positive",
                f"Confirmed (Simulated) for {d2} at 15:30 (FTX-DEMO-1002)",
                "Assign Solutions Architect and share Enterprise security & SIP trunking checklist.",
            ),
            (
                "sample-session-002",
                lead_ids["sample-session-002"],
                "Priya Nair",
                "Automated clinic appointment booking and FAQ voice assistant in English and Tamil.",
                "Matched with Starter Voice Agent ($199/month); timeline is next quarter.",
                64,
                "Warm Lead",
                "Neutral",
                f"Pending (Simulated) for {d3} at 10:00 (FTX-DEMO-1003)",
                "Send Starter Plan brochure and confirm phone number prior to demo.",
            ),
            (
                "sample-session-004",
                lead_ids["sample-session-004"],
                "Vikram Kulkarni",
                "Cost comparison between AI voice agents and manual call center staffing.",
                "Early exploration with hesitation around pricing and no active rollout timeline.",
                31,
                "Cold Lead",
                "Negative",
                "Not Scheduled",
                "Add to automated ROI case-study email sequence and follow up in 30 days.",
            ),
        ]
        for sid, lid, cname, reqs, pts, score, cat, sent, appt_st, action in sample_summaries:
            payload = {
                "customer_name": cname,
                "requirements": reqs,
                "main_points": pts,
                "lead_score": score,
                "lead_category": cat,
                "sentiment": sent,
                "appointment_status": appt_st,
                "recommended_action": action,
            }
            conn.execute(
                """
                INSERT INTO call_summaries (
                    session_id, lead_id, customer_name, requirements, main_points,
                    lead_score, lead_category, sentiment, appointment_status,
                    recommended_action, full_summary_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (sid, lid, cname, reqs, pts, score, cat, sent, appt_st, action, json.dumps(payload), _now_iso()),
            )

    return True


def upsert_lead(lead_data: Dict[str, Any], db_path: Optional[str] = None) -> int:
    """Insert or update a lead record keyed by session_id and return lead_id."""
    now = _now_iso()
    session_id = lead_data["session_id"]

    with get_connection(db_path) as conn:
        existing = conn.execute("SELECT id FROM leads WHERE session_id = ?", (session_id,)).fetchone()
        breakdown_json = json.dumps(lead_data.get("score_breakdown", {}))

        if existing:
            lead_id = int(existing["id"])
            conn.execute(
                """
                UPDATE leads SET
                    name = ?, contact = ?, email = ?, phone = ?, company = ?,
                    product_interest = ?, budget = ?, budget_amount = ?,
                    timeline = ?, requirements = ?, preferred_language = ?,
                    lead_score = ?, category = ?, score_breakdown = ?,
                    score_explanation = ?, status = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    lead_data.get("name", "Unknown") or "Unknown",
                    lead_data.get("contact", ""),
                    lead_data.get("email", ""),
                    lead_data.get("phone", ""),
                    lead_data.get("company", ""),
                    lead_data.get("product_interest", ""),
                    lead_data.get("budget", ""),
                    float(lead_data.get("budget_amount", 0) or 0),
                    lead_data.get("timeline", ""),
                    lead_data.get("requirements", ""),
                    lead_data.get("preferred_language", "English"),
                    int(lead_data.get("lead_score", 0) or 0),
                    lead_data.get("category", "Cold Lead"),
                    breakdown_json,
                    lead_data.get("score_explanation", ""),
                    lead_data.get("status", "Active"),
                    now,
                    lead_id,
                ),
            )
            return lead_id

        cur = conn.execute(
            """
            INSERT INTO leads (
                session_id, name, contact, email, phone, company,
                product_interest, budget, budget_amount, timeline, requirements,
                preferred_language, lead_score, category, score_breakdown,
                score_explanation, status, is_sample, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                session_id,
                lead_data.get("name", "Unknown") or "Unknown",
                lead_data.get("contact", ""),
                lead_data.get("email", ""),
                lead_data.get("phone", ""),
                lead_data.get("company", ""),
                lead_data.get("product_interest", ""),
                lead_data.get("budget", ""),
                float(lead_data.get("budget_amount", 0) or 0),
                lead_data.get("timeline", ""),
                lead_data.get("requirements", ""),
                lead_data.get("preferred_language", "English"),
                int(lead_data.get("lead_score", 0) or 0),
                lead_data.get("category", "Cold Lead"),
                breakdown_json,
                lead_data.get("score_explanation", ""),
                lead_data.get("status", "Active"),
                now,
                now,
            ),
        )
        return int(cur.lastrowid)


def add_conversation_turn(
    session_id: str,
    lead_id: Optional[int],
    role: str,
    message: str,
    sentiment: str = "Neutral",
    sentiment_score: float = 0.0,
    rag_sources: Optional[List[str]] = None,
    intent: str = "",
    db_path: Optional[str] = None,
) -> int:
    """Persist a single conversation message turn."""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO conversations (
                session_id, lead_id, role, message, sentiment,
                sentiment_score, rag_sources, intent, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                lead_id,
                role,
                message,
                sentiment,
                float(sentiment_score),
                json.dumps(rag_sources or []),
                intent,
                _now_iso(),
            ),
        )
        return int(cur.lastrowid)


def create_appointment_record(
    session_id: str,
    lead_id: Optional[int],
    customer_name: str,
    customer_contact: str,
    appointment_date: str,
    appointment_time: str,
    purpose: str,
    notes: str,
    confirmation_code: str,
    status: str = "Confirmed (Simulated)",
    db_path: Optional[str] = None,
) -> int:
    """Store a confirmed simulated appointment in SQLite."""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO appointments (
                session_id, lead_id, customer_name, customer_contact,
                appointment_date, appointment_time, purpose, notes,
                status, confirmation_code, is_simulated, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                session_id,
                lead_id,
                customer_name,
                customer_contact,
                appointment_date,
                appointment_time,
                purpose,
                notes,
                status,
                confirmation_code,
                _now_iso(),
            ),
        )
        return int(cur.lastrowid)


def save_call_summary(summary: Dict[str, Any], db_path: Optional[str] = None) -> int:
    """Save or update a call summary for a session."""
    with get_connection(db_path) as conn:
        existing = conn.execute(
            "SELECT id FROM call_summaries WHERE session_id = ?",
            (summary["session_id"],),
        ).fetchone()
        payload_json = json.dumps(summary)
        now = _now_iso()

        if existing:
            sid = int(existing["id"])
            conn.execute(
                """
                UPDATE call_summaries SET
                    lead_id = ?, customer_name = ?, requirements = ?, main_points = ?,
                    lead_score = ?, lead_category = ?, sentiment = ?,
                    appointment_status = ?, recommended_action = ?, full_summary_json = ?, created_at = ?
                WHERE id = ?
                """,
                (
                    summary.get("lead_id"),
                    summary.get("customer_name", "Unknown"),
                    summary.get("requirements", ""),
                    summary.get("main_points", ""),
                    int(summary.get("lead_score", 0) or 0),
                    summary.get("lead_category", "Cold Lead"),
                    summary.get("sentiment", "Neutral"),
                    summary.get("appointment_status", "Not Scheduled"),
                    summary.get("recommended_action", ""),
                    payload_json,
                    now,
                    sid,
                ),
            )
            return sid

        cur = conn.execute(
            """
            INSERT INTO call_summaries (
                session_id, lead_id, customer_name, requirements, main_points,
                lead_score, lead_category, sentiment, appointment_status,
                recommended_action, full_summary_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary["session_id"],
                summary.get("lead_id"),
                summary.get("customer_name", "Unknown"),
                summary.get("requirements", ""),
                summary.get("main_points", ""),
                int(summary.get("lead_score", 0) or 0),
                summary.get("lead_category", "Cold Lead"),
                summary.get("sentiment", "Neutral"),
                summary.get("appointment_status", "Not Scheduled"),
                summary.get("recommended_action", ""),
                payload_json,
                now,
            ),
        )
        return int(cur.lastrowid)


def record_knowledge_doc(
    filename: str,
    title: str,
    chunk_count: int,
    char_count: int,
    db_path: Optional[str] = None,
) -> None:
    """Register or update an indexed knowledge document metadata record."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO knowledge_docs (filename, title, chunk_count, char_count, uploaded_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
                title = excluded.title,
                chunk_count = excluded.chunk_count,
                char_count = excluded.char_count,
                uploaded_at = excluded.uploaded_at
            """,
            (filename, title, chunk_count, char_count, _now_iso()),
        )


def get_all_leads(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM leads ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_lead_by_session(session_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM leads WHERE session_id = ?", (session_id,)).fetchone()
        return dict(row) if row else None


def update_lead_status(lead_id: int, status: str, db_path: Optional[str] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE leads SET status = ?, updated_at = ? WHERE id = ?",
            (status, _now_iso(), lead_id),
        )


def get_conversations(session_id: Optional[str] = None, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        if session_id:
            rows = conn.execute(
                "SELECT * FROM conversations WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM conversations ORDER BY id DESC LIMIT 200").fetchall()
        return [dict(r) for r in rows]


def get_appointments(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM appointments ORDER BY appointment_date ASC, appointment_time ASC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_appointment_by_slot(
    appointment_date: str,
    appointment_time: str,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM appointments
            WHERE appointment_date = ? AND appointment_time = ? AND status NOT LIKE 'Cancelled%'
            """,
            (appointment_date, appointment_time),
        ).fetchone()
        return dict(row) if row else None


def update_appointment_status(appointment_id: int, status: str, db_path: Optional[str] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute("UPDATE appointments SET status = ? WHERE id = ?", (status, appointment_id))


def get_call_summaries(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM call_summaries ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_knowledge_docs(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM knowledge_docs ORDER BY uploaded_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_dashboard_metrics(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Compute aggregate KPIs for the Home Dashboard."""
    with get_connection(db_path) as conn:
        total_leads = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
        hot_leads = conn.execute("SELECT COUNT(*) FROM leads WHERE category = 'Hot Lead'").fetchone()[0]
        warm_leads = conn.execute("SELECT COUNT(*) FROM leads WHERE category = 'Warm Lead'").fetchone()[0]
        cold_leads = conn.execute("SELECT COUNT(*) FROM leads WHERE category = 'Cold Lead'").fetchone()[0]
        total_appointments = conn.execute(
            "SELECT COUNT(*) FROM appointments WHERE status NOT LIKE 'Cancelled%'"
        ).fetchone()[0]
        avg_score = conn.execute("SELECT COALESCE(AVG(lead_score), 0) FROM leads").fetchone()[0]
        total_summaries = conn.execute("SELECT COUNT(*) FROM call_summaries").fetchone()[0]

        sentiment_rows = conn.execute(
            """
            SELECT sentiment, COUNT(*) as cnt
            FROM conversations
            WHERE role = 'user'
            GROUP BY sentiment
            """
        ).fetchall()
        sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
        for row in sentiment_rows:
            label = row["sentiment"]
            if label in sentiment_counts:
                sentiment_counts[label] = int(row["cnt"])

        conversion_rate = round((total_appointments / total_leads * 100), 1) if total_leads else 0.0

        return {
            "total_leads": int(total_leads),
            "hot_leads": int(hot_leads),
            "warm_leads": int(warm_leads),
            "cold_leads": int(cold_leads),
            "total_appointments": int(total_appointments),
            "avg_lead_score": round(float(avg_score), 1),
            "total_summaries": int(total_summaries),
            "conversion_rate": conversion_rate,
            "sentiment_counts": sentiment_counts,
        }
