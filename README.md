# AI Log Intelligence Platform

This repository contains the foundation for a production-quality AI log analysis application built with Python, Streamlit, LangChain, Gemini, Pinecone, Plotly, and ReportLab.

## Current Status

The project foundation is initialized only. Application logic, parsers, chunking, RAG flows, Streamlit pages, analytics, and report generation are intentionally not implemented yet.

## Planned Structure

- `src/config` - Environment-driven application settings and configuration loading
- `src/core` - Future core orchestration and processing modules
- `src/models` - Shared domain models for normalized logs and analysis outputs
- `src/services` - External service wrappers for Gemini and Pinecone
- `src/ui` - Streamlit application surface
- `src/utils` - Reusable helper utilities
- `assets` - Static or generated application assets
- `reports` - Generated report outputs
- `tests` - Test suite

## Getting Started

1. Create and activate a Python environment.
2. Install dependencies from `requirements.txt`.
3. Copy `.env.example` to `.env` and add local credentials.
4. Import the package modules or extend them with implementation code.
