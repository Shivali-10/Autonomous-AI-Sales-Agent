# One-Page Project Report: Autonomous AI Sales Voice Agent (`Fortivex VoiceCRM MVP`)

## 1. Business Problem & Solution Overview
Inbound and outbound sales teams lose up to 40–60% of high-intent leads due to delayed follow-ups, inconsistent qualification, and manual CRM data entry. **Fortivex VoiceCRM MVP** solves this by deploying **Aria**, an autonomous AI sales voice and chat agent that engages prospects in real time, answers product questions grounded in verified documentation, analyzes customer sentiment, qualifies leads using a transparent 0–100 rubric, schedules validated demo slots, and logs structured call summaries directly into a SaaS CRM dashboard.

## 2. System Architecture & Technology Stack
- **Frontend & Dashboard:** Python + **Streamlit** (6-page SaaS UI: Executive Dashboard, AI Sales Agent, Lead Management, Appointments, Knowledge Base, Call Analytics) with **Plotly** visualizations and **Pandas** data tables.
- **Voice Pipeline (STT & TTS):** Browser-native `st.audio_input` WAV capture -> **SpeechRecognition** (with Gemini multimodal audio fallback) -> **gTTS** (with `pyttsx3` offline fallback) supporting English, Hindi, Hinglish, Tamil, Telugu, Spanish, French, and German.
- **LLM & Agent Orchestration:** **Google Gemini API (`google-genai`)** with deterministic RAG-grounded offline fallback (`agent.py` + `llm_service.py`) so the end-to-end flow never breaks during network or quota disruptions.
- **RAG Knowledge Retrieval:** Paragraph/sentence chunking with overlap -> **sentence-transformers (`all-MiniLM-L6-v2`)** embeddings (with offline TF-IDF fallback) -> **FAISS (`IndexFlatIP`)** cosine similarity search (`rag.py`).
- **CRM Persistence:** Auto-initialized **SQLite** database (`database.py`) storing leads, conversation turns, simulated appointments, call summaries, and indexed knowledge metadata.

## 3. Core AI Capabilities
- **Contextual Memory:** Persists customer name, email, phone, company, product interest, budget, timeline, and requirements across turns without overwriting prior facts.
- **Explainable Lead Scoring (0–100):** Weighted scoring across Budget (25 pts), Timeline (25 pts), Requirements Clarity (20 pts), Contact Completeness (15 pts), and Sentiment/Engagement (15 pts) classifying leads into **Hot (70–100)**, **Warm (40–69)**, and **Cold (0–39)** with human-readable explanations.
- **Sentiment-Adaptive Dialog:** Real-time polarity scoring (`Positive`, `Neutral`, `Negative`) dynamically adjusts Aria's tone (empathetic de-escalation for objections; closing momentum for positive signals).
- **Appointment Validation & Call Summarization:** Parses natural dates/times (e.g., *"tomorrow at 3:00 PM"*), checks Mon–Sat 09:00–19:00 business hours and slot conflicts, issues simulated confirmation codes (`FTX-YYYYMMDD-XXXX`), and generates structured post-call CRM summaries.

## 4. Scalability (100,000 Users), Security & Commercial Feasibility
- **Scaling to 100k Users:** Transition from single-node Streamlit + SQLite to **FastAPI + React/Next.js**, stateless Kubernetes workers behind an API gateway, **WebRTC/SIP streaming** (Deepgram Nova + ElevenLabs Turbo for <500ms latency), **PostgreSQL + pgvector / Qdrant** for multi-tenant RAG, and **Redis** for sub-10ms session memory.
- **Security & Compliance:** TLS 1.3 in transit, AES-256 at rest, tenant isolation (`tenant_id`), PII redaction before LLM calls, RBAC, and `.env` secret management.
- **Unit Economics & Pricing:** Estimated variable COGS of ~$0.03–$0.06/minute at scale using tiered LLM routing (Flash/Mini models) and streaming TTS, yielding >70% gross margins on Starter ($199/mo), Growth ($599/mo), and Enterprise ($1,999+/mo) plans.
