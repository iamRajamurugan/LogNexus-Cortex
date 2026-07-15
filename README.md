# LogNexus Cortex

> **Context-Aware Distributed Log Intelligence & Failure Attribution Platform**

LogNexus Cortex is an AI-powered enterprise log intelligence platform designed to transform large-scale distributed application logs into actionable operational intelligence.

Unlike conventional log analyzers that rely on keyword matching or regular expression filtering, LogNexus Cortex combines Retrieval-Augmented Generation (RAG), semantic vector search, contextual event correlation, and large language models to perform intelligent log understanding, evidence-grounded root cause inference, incident timeline reconstruction, and interactive failure investigation.

The platform enables developers, DevOps engineers, Site Reliability Engineers (SREs), and support teams to diagnose production incidents significantly faster by providing contextual, explainable, and evidence-backed insights directly from raw application logs.

---

## Key Features

- Context-aware semantic log retrieval using Retrieval-Augmented Generation (RAG)
- Intelligent parsing of enterprise application logs
- Spring Boot, Generic Log, TXT and JSON log support
- Semantic vector search powered by Gemini Embeddings
- Enterprise-scale vector indexing using Pinecone
- AI-assisted Root Cause Analysis (RCA)
- Distributed event correlation
- Incident timeline reconstruction
- Context-aware error explanation
- Interactive AI chat over uploaded log files
- Plotly-powered operational analytics dashboard
- Professional PDF incident report generation
- Modular and extensible architecture designed for enterprise-scale enhancements

---

## System Architecture

```text
                ┌────────────────────────────┐
                │      Upload Log Files      │
                └─────────────┬──────────────┘
                              │
                              ▼
                    Log Parsing Engine
                              │
                              ▼
                  Log Normalization Layer
                              │
                              ▼
                 Context-Aware Chunking
                              │
                              ▼
                Gemini Embedding Generation
                              │
                              ▼
                     Pinecone Vector Store
                              │
                              ▼
                 Semantic Retrieval Engine
                              │
                              ▼
              Context Assembly & Prompting
                              │
                              ▼
                 Gemini 2.5 Flash Reasoning
                              │
             ┌────────────────┴─────────────────┐
             ▼                                  ▼
     Root Cause Analysis               Incident Timeline
             ▼                                  ▼
      Interactive Chat             Analytics Dashboard
                              │
                              ▼
                     Professional PDF Report
```

---

## Technology Stack

### AI & LLM

- Google Gemini 2.5 Flash
- Gemini Embedding Model
- LangChain

### Vector Database

- Pinecone

### Backend

- Python
- Streamlit

### Data Processing

- Pydantic
- Plotly
- ReportLab

---

## Core Capabilities

### Semantic Log Intelligence

Retrieves operationally relevant log events using semantic similarity instead of keyword matching.

---

### Failure Attribution

Correlates distributed log events across services to identify the most probable source of production failures.

---

### Context-Aware Reasoning

Retrieves surrounding operational context before generating responses, significantly reducing hallucinations compared to traditional LLM chat systems.

---

### Incident Timeline Reconstruction

Automatically reconstructs the chronological sequence of events leading to production failures.

---

### AI-powered Root Cause Analysis

Analyzes evidence across multiple services and explains probable failure causes using grounded reasoning.

---

### Explainable AI Responses

Every generated response is backed by retrieved log evidence instead of unsupported model assumptions.

---

## Current Version

Version 1 focuses on offline enterprise log investigation.

Supported Features

- Upload log files
- Parse and normalize logs
- Intelligent event chunking
- Semantic embedding generation
- Pinecone indexing
- AI chat over logs
- Root Cause Analysis
- Timeline reconstruction
- Error explanation
- Interactive analytics
- PDF report generation

---

## Future Roadmap

- Live log ingestion
- Kubernetes log integration
- ELK integration
- Splunk integration
- OpenTelemetry support
- Grafana integration
- Multi-agent incident investigation
- Automated anomaly detection
- Cross-service dependency visualization
- Multi-session workspace
- Enterprise authentication
- Role-based access control

---

## Repository Structure

```
src/
│
├── config/
├── core/
├── models/
├── prompts/
├── services/
├── ui/
├── utils/
│
├── assets/
├── reports/
│
└── tests/
```

---

## Project Vision

Modern enterprise systems generate millions of log events every day.

Traditional log analysis workflows require engineers to manually search, correlate, and interpret thousands of log lines before identifying the true source of a production incident.

LogNexus Cortex aims to bridge this gap by combining semantic retrieval, contextual reasoning, and evidence-grounded AI to accelerate production debugging while maintaining explainability and developer trust.

---

## Author

**Rajamurugan A**

Artificial Intelligence & Machine Learning Engineer

---

## License

This project is intended for educational, research, and portfolio purposes.
