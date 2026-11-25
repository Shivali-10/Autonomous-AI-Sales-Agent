"""Streamlit SaaS Dashboard for the Autonomous AI Sales Voice Agent."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from agent import SalesVoiceAgent
from appointment import book_appointment
from database import (
    get_all_leads,
    get_appointments,
    get_call_summaries,
    get_conversations,
    get_dashboard_metrics,
    get_knowledge_docs,
    init_db,
    seed_sample_data,
    update_appointment_status,
    update_lead_status,
)
from lead_scoring import score_lead
from llm_service import LLMService
from rag import KnowledgeBaseRAG
from summarizer import generate_call_summary
from voice_service import LANGUAGE_CODES, synthesize_speech_bytes, transcribe_audio_bytes


load_dotenv()

st.set_page_config(
    page_title="Fortivex AI | Autonomous Sales Voice Agent",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    .saas-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 65%, #2563EB 100%);
        color: #FFFFFF;
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
    }
    .saas-header h1 { margin: 0; font-size: 1.65rem; font-weight: 700; color: #FFFFFF; }
    .saas-header p { margin: 0.35rem 0 0 0; font-size: 0.92rem; color: #CBD5E1; }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem 1.15rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .kpi-label { font-size: 0.8rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.04em; }
    .kpi-value { font-size: 1.75rem; font-weight: 700; color: #0F172A; margin-top: 0.25rem; }
    .kpi-sub { font-size: 0.8rem; color: #475569; margin-top: 0.2rem; }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-hot { background: #FEE2E2; color: #B91C1C; }
    .badge-warm { background: #FEF3C7; color: #B45309; }
    .badge-cold { background: #DBEAFE; color: #1D4ED8; }
    .badge-pos { background: #DCFCE7; color: #15803D; }
    .badge-neu { background: #F1F5F9; color: #334155; }
    .badge-neg { background: #FEE2E2; color: #B91C1C; }
    .panel-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.9rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_services():
    init_db()
    rag = KnowledgeBaseRAG()
    llm = LLMService()
    agent = SalesVoiceAgent(rag=rag, llm=llm)
    return rag, llm, agent


rag_service, llm_service, sales_agent = get_services()


def _init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"live-{uuid.uuid4().hex[:8]}"
    if "language" not in st.session_state:
        st.session_state.language = "English"
    if "lead_profile" not in st.session_state:
        st.session_state.lead_profile = SalesVoiceAgent.default_profile(
            st.session_state.session_id, st.session_state.language
        )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "message": (
                    "Hello! I'm Aria, your Autonomous AI Sales Representative at Fortivex AI. "
                    "I can answer questions about our Voice AI & CRM plans, understand your requirements, "
                    "and schedule a product demo for you. How can I help your business today?"
                ),
                "sentiment": "Neutral",
                "rag_sources": [],
            }
        ]
    if "latest_score" not in st.session_state:
        st.session_state.latest_score = score_lead(st.session_state.lead_profile).to_dict()
    if "latest_sentiment" not in st.session_state:
        st.session_state.latest_sentiment = {
            "label": "Neutral",
            "score": 0.0,
            "confidence": 0.6,
            "cues": [],
            "tone_guidance": "Friendly, consultative sales tone.",
        }
    if "latest_rag_chunks" not in st.session_state:
        st.session_state.latest_rag_chunks = []
    if "session_appointment" not in st.session_state:
        st.session_state.session_appointment = None
    if "latest_summary" not in st.session_state:
        st.session_state.latest_summary = None
    if "last_tts_audio" not in st.session_state:
        st.session_state.last_tts_audio = None
    if "processed_audio_hash" not in st.session_state:
        st.session_state.processed_audio_hash = None


_init_session_state()


def _reset_live_conversation() -> None:
    st.session_state.session_id = f"live-{uuid.uuid4().hex[:8]}"
    st.session_state.lead_profile = SalesVoiceAgent.default_profile(
        st.session_state.session_id, st.session_state.language
    )
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "message": (
                "Hello! I'm Aria, your Autonomous AI Sales Representative at Fortivex AI. "
                "How can I help your team automate sales calls or customer engagement today?"
            ),
            "sentiment": "Neutral",
            "rag_sources": [],
        }
    ]
    st.session_state.latest_score = score_lead(st.session_state.lead_profile).to_dict()
    st.session_state.latest_sentiment = {
        "label": "Neutral",
        "score": 0.0,
        "confidence": 0.6,
        "cues": [],
        "tone_guidance": "Friendly, consultative sales tone.",
    }
    st.session_state.latest_rag_chunks = []
    st.session_state.session_appointment = None
    st.session_state.latest_summary = None
    st.session_state.last_tts_audio = None
    st.session_state.processed_audio_hash = None


def _category_badge(cat: str) -> str:
    if cat == "Hot Lead":
        return '<span class="badge badge-hot">Hot Lead</span>'
    if cat == "Warm Lead":
        return '<span class="badge badge-warm">Warm Lead</span>'
    return '<span class="badge badge-cold">Cold Lead</span>'


def _sentiment_badge(sent: str) -> str:
    if sent == "Positive":
        return '<span class="badge badge-pos">Positive</span>'
    if sent == "Negative":
        return '<span class="badge badge-neg">Negative</span>'
    return '<span class="badge badge-neu">Neutral</span>'


def _render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🎙️ Fortivex VoiceCRM")
        st.caption("Autonomous AI Sales Voice Agent MVP")

        page = st.radio(
            "Navigation",
            [
                "Home Dashboard",
                "AI Sales Agent",
                "Lead Management",
                "Appointments",
                "Knowledge Base",
                "Call Analytics",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("### System Status")
        if llm_service.is_live_llm_available:
            st.success(f"LLM: {llm_service.provider_status}")
        else:
            st.warning("LLM: Local Fallback Mode (Add GEMINI_API_KEY in .env for live LLM)")

        st.info(f"RAG: {rag_service.backend_mode}")
        st.caption("CRM & Bookings: Simulated Local SQLite")

        st.divider()
        if st.button("Reset Demo Sample Data", use_container_width=True):
            seed_sample_data(force=True)
            st.success("Sample CRM data restored.")
            st.rerun()

    return page


def render_home_dashboard() -> None:
    st.markdown(
        """
        <div class="saas-header">
            <h1>Autonomous AI Sales Voice Agent — Executive Dashboard</h1>
            <p>Real-time overview of AI-qualified leads, sentiment trends, and simulated demo bookings stored in SQLite.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = get_dashboard_metrics()
    leads = get_all_leads()
    appointments = get_appointments()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Total Leads</div><div class="kpi-value">{metrics["total_leads"]}</div><div class="kpi-sub">Avg Score: {metrics["avg_lead_score"]}/100</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Hot Leads</div><div class="kpi-value" style="color:#DC2626;">{metrics["hot_leads"]}</div><div class="kpi-sub">Score ≥ 70</div></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Warm / Cold</div><div class="kpi-value">{metrics["warm_leads"]} / {metrics["cold_leads"]}</div><div class="kpi-sub">Pipeline Segmentation</div></div>',
        unsafe_allow_html=True,
    )
    c4.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Appointments</div><div class="kpi-value" style="color:#2563EB;">{metrics["total_appointments"]}</div><div class="kpi-sub">Simulated SQLite Bookings</div></div>',
        unsafe_allow_html=True,
    )
    c5.markdown(
        f'<div class="kpi-card"><div class="kpi-label">Demo Conversion</div><div class="kpi-value" style="color:#16A34A;">{metrics["conversion_rate"]}%</div><div class="kpi-sub">{metrics["total_summaries"]} Call Summaries</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Lead Qualification Breakdown")
        cat_df = pd.DataFrame(
            {
                "Category": ["Hot Lead", "Warm Lead", "Cold Lead"],
                "Count": [metrics["hot_leads"], metrics["warm_leads"], metrics["cold_leads"]],
            }
        )
        fig_cat = px.bar(
            cat_df,
            x="Category",
            y="Count",
            color="Category",
            color_discrete_map={
                "Hot Lead": "#EF4444",
                "Warm Lead": "#F59E0B",
                "Cold Lead": "#3B82F6",
            },
            text="Count",
        )
        fig_cat.update_layout(showlegend=False, height=320, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_r:
        st.subheader("Customer Sentiment Distribution")
        s_counts = metrics["sentiment_counts"]
        sent_df = pd.DataFrame(
            {
                "Sentiment": list(s_counts.keys()),
                "Messages": list(s_counts.values()),
            }
        )
        fig_sent = px.pie(
            sent_df,
            names="Sentiment",
            values="Messages",
            hole=0.45,
            color="Sentiment",
            color_discrete_map={
                "Positive": "#22C55E",
                "Neutral": "#94A3B8",
                "Negative": "#EF4444",
            },
        )
        fig_sent.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_sent, use_container_width=True)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.subheader("Recent Lead Activity")
        if leads:
            df_leads = pd.DataFrame(leads)[
                ["name", "company", "product_interest", "lead_score", "category", "status", "updated_at"]
            ]
            st.dataframe(df_leads, use_container_width=True, hide_index=True)
        else:
            st.info("No leads captured yet. Start a conversation on the AI Sales Agent page.")

    with col_b:
        st.subheader("Upcoming Simulated Appointments")
        if appointments:
            df_appt = pd.DataFrame(appointments)[
                ["customer_name", "appointment_date", "appointment_time", "status", "confirmation_code"]
            ]
            st.dataframe(df_appt, use_container_width=True, hide_index=True)
        else:
            st.info("No appointments booked yet.")


def _handle_customer_message(user_text: str, enable_tts: bool) -> None:
    if not user_text or not user_text.strip():
        return

    result = sales_agent.process_turn(
        session_id=st.session_state.session_id,
        user_message=user_text.strip(),
        lead_profile=st.session_state.lead_profile,
        history=st.session_state.chat_history,
        language=st.session_state.language,
        existing_appointment=st.session_state.session_appointment,
    )

    st.session_state.chat_history.append(
        {
            "role": "user",
            "message": user_text.strip(),
            "sentiment": result["sentiment"]["label"],
            "rag_sources": result["rag_sources"],
        }
    )
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "message": result["reply"],
            "sentiment": "Neutral",
            "rag_sources": result["rag_sources"],
        }
    )

    st.session_state.lead_profile = result["lead_profile"]
    st.session_state.latest_score = result["lead_score"]
    st.session_state.latest_sentiment = result["sentiment"]
    st.session_state.latest_rag_chunks = result["rag_chunks"]
    if result["appointment"]:
        st.session_state.session_appointment = result["appointment"]

    if enable_tts:
        audio_bytes, tts_err = synthesize_speech_bytes(result["reply"], language=st.session_state.language)
        st.session_state.last_tts_audio = audio_bytes
        if tts_err:
            st.toast(tts_err, icon="ℹ️")
    else:
        st.session_state.last_tts_audio = None


def render_ai_sales_agent() -> None:
    st.markdown(
        """
        <div class="saas-header">
            <h1>AI Sales Voice Agent (Aria)</h1>
            <p>Speak via microphone or chat via text. Aria qualifies leads, adapts to sentiment, retrieves RAG knowledge, and books simulated demos.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_c1, top_c2, top_c3, top_c4 = st.columns([2, 2, 2, 2])
    with top_c1:
        st.session_state.language = st.selectbox(
            "Conversation Language",
            list(LANGUAGE_CODES.keys()),
            index=list(LANGUAGE_CODES.keys()).index(st.session_state.language),
        )
    with top_c2:
        enable_tts = st.toggle("Enable Voice Reply (TTS)", value=True)
    with top_c3:
        st.caption(f"Session ID: `{st.session_state.session_id}`")
    with top_c4:
        if st.button("Start New Call Session", use_container_width=True):
            _reset_live_conversation()
            st.rerun()

    left_col, right_col = st.columns([1.45, 1.0], gap="large")

    with left_col:
        st.subheader("Voice & Text Conversation")

        # Voice Input Widget (Browser-native Streamlit mic capture -> STT)
        with st.expander("🎙️ Voice Input (Record Microphone or Upload WAV)", expanded=True):
            audio_value = st.audio_input("Click the microphone to speak to Aria")
            if audio_value is not None:
                raw_bytes = audio_value.getvalue()
                audio_hash = hash(raw_bytes)
                if audio_hash != st.session_state.processed_audio_hash:
                    st.session_state.processed_audio_hash = audio_hash
                    with st.spinner("Transcribing speech (STT)..."):
                        transcript, stt_err = transcribe_audio_bytes(
                            raw_bytes,
                            language=st.session_state.language,
                            llm_service=llm_service,
                        )
                    if transcript:
                        st.success(f"Heard: \"{transcript}\"")
                        with st.spinner("Aria is thinking & updating CRM..."):
                            _handle_customer_message(transcript, enable_tts=enable_tts)
                        st.rerun()
                    else:
                        st.warning(stt_err or "Could not transcribe audio. Please use text chat below.")

        # Conversation History Display
        chat_box = st.container(height=390, border=True)
        with chat_box:
            for turn in st.session_state.chat_history:
                role = turn["role"]
                with st.chat_message("user" if role == "user" else "assistant", avatar="🧑" if role == "user" else "🤖"):
                    st.markdown(turn["message"])
                    if role == "user":
                        st.caption(f"Detected Sentiment: {turn.get('sentiment', 'Neutral')}")
                    elif turn.get("rag_sources"):
                        st.caption(f"Grounded in: {', '.join(turn['rag_sources'])}")

        if st.session_state.last_tts_audio:
            st.markdown("**Latest Agent Voice Response (TTS):**")
            st.audio(st.session_state.last_tts_audio, format="audio/mp3", autoplay=True)

        # Quick Demo Prompt Helpers for smooth interview demonstration
        st.caption("Quick Demo Scenarios (click to simulate a customer message):")
        q1, q2, q3 = st.columns(3)
        if q1.button("1. Inquiry + Pricing", use_container_width=True):
            _handle_customer_message(
                "Hi, I'm Ananya from Zenith Health. We need an English & Hindi voice agent for 3,000 calls/month. What plans and pricing do you offer?",
                enable_tts=enable_tts,
            )
            st.rerun()
        if q2.button("2. Budget & Timeline", use_container_width=True):
            _handle_customer_message(
                "Growth Pro sounds great! Our budget is $600/month, we want to launch this month, and you can reach me at ananya@zenithhealth.in or 9876501234.",
                enable_tts=enable_tts,
            )
            st.rerun()
        if q3.button("3. Book Demo Slot", use_container_width=True):
            _handle_customer_message(
                "Please book a demo for tomorrow at 3:00 PM.",
                enable_tts=enable_tts,
            )
            st.rerun()

        typed_msg = st.chat_input("Type your message here (text chat fallback)...")
        if typed_msg:
            with st.spinner("Processing message, RAG retrieval, and lead scoring..."):
                _handle_customer_message(typed_msg, enable_tts=enable_tts)
            st.rerun()

    with right_col:
        st.subheader("Live AI & CRM Intelligence")

        # Lead Score Card
        score_data = st.session_state.latest_score
        score_val = score_data.get("score", 0)
        cat_val = score_data.get("category", "Cold Lead")
        st.markdown(
            f'<div class="panel-box"><strong>Lead Qualification Score:</strong> {score_val}/100 &nbsp; {_category_badge(cat_val)}',
            unsafe_allow_html=True,
        )
        st.progress(min(100, max(0, int(score_val))) / 100.0)
        st.caption(score_data.get("explanation", ""))
        breakdown = score_data.get("breakdown", {})
        if breakdown:
            b_cols = st.columns(5)
            b_cols[0].metric("Budget", f"{breakdown.get('budget', 0)}/25")
            b_cols[1].metric("Timeline", f"{breakdown.get('timeline', 0)}/25")
            b_cols[2].metric("Needs", f"{breakdown.get('requirements', 0)}/20")
            b_cols[3].metric("Contact", f"{breakdown.get('contact', 0)}/15")
            b_cols[4].metric("Signal", f"{breakdown.get('sentiment', 0)}/15")
        st.markdown("</div>", unsafe_allow_html=True)

        # Sentiment Card
        sent_data = st.session_state.latest_sentiment
        st.markdown(
            f'<div class="panel-box"><strong>Customer Sentiment:</strong> &nbsp; {_sentiment_badge(sent_data["label"])} (Score: {sent_data["score"]})<br><small style="color:#475569;"><strong>Agent Adaptation:</strong> {sent_data["tone_guidance"]}</small></div>',
            unsafe_allow_html=True,
        )

        # Contextual Memory Profile
        prof = st.session_state.lead_profile
        with st.expander("🧠 Contextual Lead Memory", expanded=True):
            st.markdown(
                f"""
                - **Name:** {prof.get('name') or 'Not shared'}
                - **Company:** {prof.get('company') or 'Not shared'}
                - **Contact:** {prof.get('contact') or 'Not shared'}
                - **Product Interest:** {prof.get('product_interest') or 'TBD'}
                - **Budget:** {prof.get('budget') or 'TBD'}
                - **Timeline:** {prof.get('timeline') or 'TBD'}
                - **Requirements:** {prof.get('requirements') or 'Gathering...'}
                """
            )

        # Appointment Status / Quick Booking
        with st.expander("📅 Appointment Booking (Simulated CRM)", expanded=True):
            appt = st.session_state.session_appointment
            if appt and appt.get("confirmation_code"):
                st.success(
                    f"Confirmed (Simulated): {appt['appointment_date']} at {appt['appointment_time']} "
                    f"| Ref: `{appt['confirmation_code']}`"
                )
            else:
                st.caption("Ask Aria in chat/voice to book a slot, or confirm manually below:")
                with st.form("quick_appt_form", clear_on_submit=False):
                    default_dt = (datetime.now() + timedelta(days=1)).date()
                    d_pick = st.date_input("Date", value=default_dt)
                    t_pick = st.text_input("Time (HH:MM 24h)", value="15:00")
                    submitted = st.form_submit_button("Confirm Simulated Booking")
                    if submitted:
                        res = book_appointment(
                            session_id=st.session_state.session_id,
                            lead_id=prof.get("id"),
                            customer_name=prof.get("name", ""),
                            customer_contact=prof.get("contact", ""),
                            appointment_date=d_pick.strftime("%Y-%m-%d"),
                            appointment_time=t_pick,
                            purpose=f"Demo: {prof.get('product_interest') or 'Fortivex AI'}",
                            notes=prof.get("requirements", ""),
                        )
                        if res.success:
                            st.session_state.session_appointment = res.to_dict()
                            st.success(res.message)
                            st.rerun()
                        else:
                            st.error(res.message)

        # RAG Retrieved Context
        with st.expander("📚 Retrieved RAG Knowledge Chunks", expanded=False):
            if st.session_state.latest_rag_chunks:
                for ch in st.session_state.latest_rag_chunks:
                    st.markdown(f"**{ch['source']}** (sim: `{ch['score']:.3f}`)")
                    st.caption(ch["text"][:260] + ("..." if len(ch["text"]) > 260 else ""))
            else:
                st.caption("Ask a product, pricing, or security question to view FAISS vector matches.")

        # End Call & Summarize
        st.divider()
        if st.button("📝 End Call & Generate Summary", type="primary", use_container_width=True):
            summary = generate_call_summary(
                session_id=st.session_state.session_id,
                lead_id=prof.get("id"),
                lead_profile=prof,
                score_result=st.session_state.latest_score,
                messages=st.session_state.chat_history,
                appointment_info=st.session_state.session_appointment,
                llm_service=llm_service,
                persist=True,
            )
            st.session_state.latest_summary = summary
            st.success("Call summary generated and saved to SQLite CRM!")

        if st.session_state.latest_summary:
            s = st.session_state.latest_summary
            with st.container(border=True):
                st.markdown("#### Saved Call Summary")
                st.markdown(f"- **Customer:** {s['customer_name']}")
                st.markdown(f"- **Lead Score:** {s['lead_score']}/100 ({s['lead_category']})")
                st.markdown(f"- **Sentiment:** {s['sentiment']}")
                st.markdown(f"- **Appointment:** {s['appointment_status']}")
                st.markdown(f"- **Requirements:** {s['requirements']}")
                st.markdown(f"- **Discussion Points:** {s['main_points']}")
                st.markdown(f"- **Recommended Action:** {s['recommended_action']}")


def render_lead_management() -> None:
    st.markdown(
        """
        <div class="saas-header">
            <h1>Lead Management (CRM Pipeline)</h1>
            <p>Inspect AI-qualified leads, score breakdowns, explanations, and pipeline status stored in SQLite.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    leads = get_all_leads()
    if not leads:
        st.info("No leads in database yet.")
        return

    df = pd.DataFrame(leads)

    f1, f2, f3 = st.columns([2, 2, 1])
    cat_filter = f1.multiselect(
        "Filter by Lead Category",
        ["Hot Lead", "Warm Lead", "Cold Lead"],
        default=["Hot Lead", "Warm Lead", "Cold Lead"],
    )
    search_q = f2.text_input("Search by Name, Company, or Product")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    f3.download_button("Export CSV", data=csv_bytes, file_name="fortivex_leads.csv", mime="text/csv", use_container_width=True)

    filtered = df[df["category"].isin(cat_filter)]
    if search_q.strip():
        q = search_q.strip().lower()
        filtered = filtered[
            filtered["name"].str.lower().str.contains(q, na=False)
            | filtered["company"].str.lower().str.contains(q, na=False)
            | filtered["product_interest"].str.lower().str.contains(q, na=False)
        ]

    st.dataframe(
        filtered[
            ["id", "name", "company", "contact", "product_interest", "budget", "timeline", "lead_score", "category", "status", "updated_at"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Inspect & Update Lead")
    lead_options = {f"#{row['id']} - {row['name']} ({row['category']}, {row['lead_score']}/100)": row for row in leads}
    selected_label = st.selectbox("Select a Lead", list(lead_options.keys()))
    selected = lead_options[selected_label]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Score Explanation:** {selected['score_explanation']}")
        st.markdown(f"**Captured Requirements:** {selected['requirements'] or 'None recorded'}")
        try:
            bd = json.loads(selected.get("score_breakdown") or "{}")
            if bd:
                fig = go.Figure(
                    go.Bar(
                        x=list(bd.values()),
                        y=[k.title() for k in bd.keys()],
                        orientation="h",
                        marker_color="#2563EB",
                    )
                )
                fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Points")
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    with col2:
        statuses = ["New", "Active", "Qualified", "Demo Scheduled", "Proposal Sent", "Nurture", "Closed Won"]
        cur_status = selected["status"] if selected["status"] in statuses else "Active"
        new_status = st.selectbox("Pipeline Status", statuses, index=statuses.index(cur_status))
        if st.button("Update Status", type="primary", use_container_width=True):
            update_lead_status(int(selected["id"]), new_status)
            st.success(f"Updated status for {selected['name']} to '{new_status}'.")
            st.rerun()


def render_appointments() -> None:
    st.markdown(
        """
        <div class="saas-header">
            <h1>Appointments & Demo Schedule</h1>
            <p>Manage simulated CRM appointments validated for business hours and slot conflicts in SQLite.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Note: Appointments shown here are simulated local SQLite CRM records for MVP demonstration.")

    appts = get_appointments()
    c_left, c_right = st.columns([1.8, 1.0], gap="large")

    with c_left:
        st.subheader("Booked Appointments")
        if appts:
            df = pd.DataFrame(appts)
            st.dataframe(
                df[
                    ["id", "confirmation_code", "customer_name", "customer_contact", "appointment_date", "appointment_time", "purpose", "status"]
                ],
                use_container_width=True,
                hide_index=True,
            )

            appt_map = {
                f"{a['confirmation_code']} | {a['customer_name']} ({a['appointment_date']} {a['appointment_time']})": a
                for a in appts
            }
            sel = st.selectbox("Update Appointment Status", list(appt_map.keys()))
            target = appt_map[sel]
            u1, u2 = st.columns(2)
            if u1.button("Mark Completed (Simulated)", use_container_width=True):
                update_appointment_status(int(target["id"]), "Completed (Simulated)")
                st.rerun()
            if u2.button("Cancel Appointment", use_container_width=True):
                update_appointment_status(int(target["id"]), "Cancelled")
                st.rerun()
        else:
            st.info("No appointments booked yet.")

    with c_right:
        st.subheader("Schedule New Simulated Demo")
        with st.form("manual_booking_form"):
            cname = st.text_input("Customer Name *")
            ccontact = st.text_input("Email or Phone *")
            default_dt = (datetime.now() + timedelta(days=1)).date()
            adate = st.date_input("Preferred Date *", value=default_dt)
            atime = st.text_input("Preferred Time (HH:MM 24h, 09:00-18:59) *", value="14:00")
            purpose = st.text_input("Purpose", value="Product Demo & Consultation")
            notes = st.text_area("Notes", height=80)
            if st.form_submit_button("Book Simulated Appointment", type="primary", use_container_width=True):
                res = book_appointment(
                    session_id=f"manual-{uuid.uuid4().hex[:6]}",
                    lead_id=None,
                    customer_name=cname,
                    customer_contact=ccontact,
                    appointment_date=adate.strftime("%Y-%m-%d"),
                    appointment_time=atime,
                    purpose=purpose,
                    notes=notes,
                )
                if res.success:
                    st.success(res.message)
                    st.rerun()
                else:
                    st.error(res.message)


def render_knowledge_base() -> None:
    st.markdown(
        """
        <div class="saas-header">
            <h1>RAG Knowledge Base</h1>
            <p>Upload company documents (.txt, .md, .pdf, .csv, .json) to chunk, embed, and index in FAISS for grounded AI answers.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.1, 1.2], gap="large")
    with c1:
        st.subheader("Upload & Index Document")
        uploaded = st.file_uploader(
            "Choose a knowledge document",
            type=["txt", "md", "pdf", "csv", "json"],
        )
        if uploaded is not None and st.button("Ingest & Rebuild FAISS Index", type="primary"):
            with st.spinner("Chunking document and building FAISS embeddings..."):
                total_chunks = rag_service.add_uploaded_file(uploaded.name, uploaded.getvalue())
            st.success(f"Indexed '{uploaded.name}'. Total chunks in FAISS index: {total_chunks}")
            st.rerun()

        if st.button("Force Rebuild Index from Disk"):
            total_chunks = rag_service.rebuild_index()
            st.success(f"Rebuilt FAISS index with {total_chunks} chunks ({rag_service.backend_mode}).")

        st.subheader("Indexed Documents")
        docs = get_knowledge_docs()
        if docs:
            st.dataframe(pd.DataFrame(docs), use_container_width=True, hide_index=True)
        else:
            st.info("No documents indexed.")

    with c2:
        st.subheader("Test Vector Retrieval (RAG Inspector)")
        st.caption(f"Active Vector Backend: `{rag_service.backend_mode}`")
        query = st.text_input("Enter a customer question to test retrieval", value="What is the price of the Growth plan and does it support Hindi?")
        top_k = st.slider("Top-K Chunks", min_value=1, max_value=6, value=3)
        if query.strip():
            hits = rag_service.search(query, top_k=top_k)
            if hits:
                for idx, hit in enumerate(hits, start=1):
                    with st.container(border=True):
                        st.markdown(f"**#{idx} | {hit.source}** (chunk {hit.chunk_id}, similarity `{hit.score:.4f}`)")
                        st.write(hit.text)
            else:
                st.warning("No matching chunks found.")


def render_call_analytics() -> None:
    st.markdown(
        """
        <div class="saas-header">
            <h1>Call Analytics & Conversation Summaries</h1>
            <p>Review automated post-call summaries, sentiment analytics, and full session transcripts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summaries = get_call_summaries()
    conversations = get_conversations()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sentiment Across Customer Turns")
        if conversations:
            conv_df = pd.DataFrame(conversations)
            user_df = conv_df[conv_df["role"] == "user"]
            if not user_df.empty:
                counts = user_df["sentiment"].value_counts().reset_index()
                counts.columns = ["Sentiment", "Count"]
                fig = px.bar(
                    counts,
                    x="Sentiment",
                    y="Count",
                    color="Sentiment",
                    color_discrete_map={"Positive": "#22C55E", "Neutral": "#94A3B8", "Negative": "#EF4444"},
                )
                fig.update_layout(showlegend=False, height=290, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No conversation turns logged yet.")

    with c2:
        st.subheader("Lead Score Distribution")
        leads = get_all_leads()
        if leads:
            ldf = pd.DataFrame(leads)
            fig_hist = px.histogram(ldf, x="lead_score", nbins=10, color="category",
                                    color_discrete_map={"Hot Lead": "#EF4444", "Warm Lead": "#F59E0B", "Cold Lead": "#3B82F6"})
            fig_hist.update_layout(height=290, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Lead Score (0-100)")
            st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Saved Call Summaries")
    if summaries:
        sum_df = pd.DataFrame(summaries)
        st.download_button(
            "Download Call Summaries CSV",
            data=sum_df.to_csv(index=False).encode("utf-8"),
            file_name="fortivex_call_summaries.csv",
            mime="text/csv",
        )
        for s in summaries:
            with st.expander(
                f"{s['customer_name']} | Score: {s['lead_score']} ({s['lead_category']}) | Sentiment: {s['sentiment']} | {s['created_at']}"
            ):
                st.markdown(f"- **Requirements:** {s['requirements']}")
                st.markdown(f"- **Main Discussion Points:** {s['main_points']}")
                st.markdown(f"- **Appointment Status:** {s['appointment_status']}")
                st.markdown(f"- **Recommended Next Action:** {s['recommended_action']}")

                session_turns = get_conversations(session_id=s["session_id"])
                if session_turns:
                    st.markdown("**Transcript:**")
                    for t in session_turns:
                        st.caption(f"[{t['created_at']}] **{t['role'].upper()}** ({t['sentiment']}): {t['message']}")
    else:
        st.info("No call summaries generated yet. Use 'End Call & Generate Summary' in AI Sales Agent.")


selected_page = _render_sidebar()

if selected_page == "Home Dashboard":
    render_home_dashboard()
elif selected_page == "AI Sales Agent":
    render_ai_sales_agent()
elif selected_page == "Lead Management":
    render_lead_management()
elif selected_page == "Appointments":
    render_appointments()
elif selected_page == "Knowledge Base":
    render_knowledge_base()
elif selected_page == "Call Analytics":
    render_call_analytics()
