# Fortivex AI — Security, SLA & Deployment FAQ

## Data Security & Privacy
- **Encryption in Transit:** TLS 1.3 enforced across all web, WebSocket, and telephony webhooks.
- **Encryption at Rest:** AES-256 encryption for database records, uploaded RAG documents, and call summaries.
- **PII Redaction:** Automated masking of phone numbers, credit card numbers, and government IDs before LLM inference on Enterprise plans.
- **Zero-Data-Retention Mode:** Available on Enterprise plans so customer call transcripts are never used for third-party model training.

## Service Level Agreements (SLA)
- **Starter Plan:** 99.5% uptime, 24-hour email support response.
- **Growth Pro Plan:** 99.9% uptime, 4-hour priority chat & phone support.
- **Enterprise Plan:** 99.95% uptime SLA, 1-hour 24/7 response, dedicated Solutions Architect.

## Deployment Options
- **Multi-Tenant Cloud (India / Global):** Hosted in AWS `ap-south-1` (Mumbai) or `us-east-1` with auto-scaling containers.
- **Dedicated Single-Tenant VPC:** Isolated Kubernetes cluster with private vector database and VPC peering.
- **Hybrid / On-Premise:** Containerized Docker + Kubernetes Helm deployment for regulated BFSI and healthcare organizations.
