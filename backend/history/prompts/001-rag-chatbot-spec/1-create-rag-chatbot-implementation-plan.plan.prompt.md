---
id: 1
title: Create RAG Chatbot Implementation Plan
stage: plan
date: 2026-01-11
surface: agent
model: claude-sonnet-4-5-20250929
feature: 001-rag-chatbot-spec
branch: 001-rag-chatbot-spec
user: Claude
command: /sp.plan
labels: [rag, chatbot, planning, architecture]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-rag-chatbot-spec/plan.md
  - specs/001-rag-chatbot-spec/research.md
  - specs/001-rag-chatbot-spec/data-model.md
  - specs/001-rag-chatbot-spec/quickstart.md
  - specs/001-rag-chatbot-spec/contracts/chatbot-api.yaml
tests:
  - null
---

## Prompt

You are a specialized RAG assistant for the published book on Integrated RAG Chatbot Development. Your planning must be grounded exclusively in the ingested content from the my-web docs folder (indexed via Qdrant and Neon Serverless Postgres). Adhere strictly to this plan prompt, which enforces strategic planning for query handling, implementation steps, credential usage, and alignment with project requirements. Violate no part of it—internally critique every planned step against these rules before execution. Use the specified models for embedding, response generation, and answer handling: primary xiaomi/mimo-v2-flash:free, fallback mistralai/devstral-2512:free, then tngtech/deepseek-r1t2-chimera:free via OpenRouter, with automatic fallback logic to handle rate limits without interruption.
Target Audience:

Backend Developers Integrating RAG Systems: Plan responses for users focused on practical setup, emphasizing secure credential handling, API integrations, and scalable backend planning.

Focus:

Strategic Planning for RAG Operations: Center on planning ingestion, querying, error handling, and credential-based connections (e.g., Neon DB, Qdrant, OpenRouter). Include detailed steps for using provided credentials securely in code without hardcoding.

Success Criteria:

Comprehensive Planning: Every plan must outline at least 3-5 sequential steps for tasks like data ingestion, vector indexing, query retrieval, and response generation, ensuring reproducibility from book content.
Credential Integration: Incorporate exact credentials in planning examples: Neon DB connection string 'postgresql://neondb_owner:npg_DTLjYQkl12UB@ep-dark-dew-ahg8v6x1-pooler.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require'; Qdrant cluster endpoint 'https://5ae53cc1-dbc4-44ef-a5d9-9a27778a70f9.us-east4-0.gcp.cloud.qdrant.io'; Qdrant API key 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.FAwrR6glYWhLMLpx_yR4gOGN4nkRtiuNMJGFWIYe3EM'; Qdrant cluster ID '5ae53cc1-dbc4-44ef-a5d9-9a27778a70f9'; OpenRouter API key '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'. Advise using environment variables for security.
Rate Limit and Fallback Resilience: Plan must include fallback mechanisms in code snippets, ensuring seamless model switching.
Grounded in Book Content: All planning steps supported by references to ingested book sections or chunks.

Constraints:

Planning Efficiency: Keep plans concise—under 500 words, structured as numbered steps, aligning with book-embedded chatbot efficiency.
Security Awareness: Emphasize not exposing credentials in production; use .env files or secrets management in plans.
Free-Tier Adherence: Plan around Qdrant Cloud Free Tier limits (e.g., storage, queries) and Neon serverless constraints.
Tool Integration: Incorporate Claude CLI for prompt testing and Speckit for API spec validation in development plans.

Exclusions (Not Planning):

No frontend planning or UI elements.
No installation of external SDKs; stick to requests and core libs.
No paid features; enforce free models and tiers.
No unrelated expansions beyond book RAG topics.

Query Handling in Planning:

If selected text is provided, plan using ONLY that as context for embedding and generation steps.
For general questions, plan retrieval from Qdrant-indexed content, embed queries with specified models, and outline augmented prompt generation.
If info is absent: Plan to respond "Not covered in the ingested book content."
No external knowledge, opinions, or hallucinations allowed. All planning for embedding, text processing, and answer generation must use the exact models with fallback, integrating provided credentials where relevant (e.g., in connection code examples).

## Response snapshot

# Implementation Plan: [FEATURE]

**Branch**: `[###-feature-name]` | **Date**: [DATE] | **Spec**: [link]
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation plan for a RAG (Retrieval-Augmented Generation) chatbot system designed for backend developers integrating RAG systems. The system leverages Qdrant for vector storage, Neon Serverless Postgres for metadata, and OpenRouter for LLM access with fallback mechanisms. Key components include document ingestion pipeline, vector indexing, conversation management, and secure credential handling. The architecture follows a service-oriented approach with clear separation of concerns between API layer, business logic, and external service integrations.

## Technical Context

**Language/Version**: Python 3.11+ (for FastAPI chatbot), FastAPI 0.104+
**Primary Dependencies**: FastAPI, Qdrant, Cohere, Better Auth, Neon Serverless Postgres
**Storage**: Qdrant vector database for RAG chatbot, Neon Serverless Postgres for metadata
**Testing**: pytest for backend
**Target Platform**: Linux server (backend)
**Project Type**: Backend service
**Performance Goals**: <200ms p95 latency for RAG queries, handle 1000 concurrent users
**Constraints**: Free-tier adherence (Qdrant Cloud Free Tier limits, Neon serverless constraints), secure credential handling
**Scale/Scope**: Backend developers integrating RAG systems, educational content delivery

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Gate 1: Accuracy through Primary Source Verification
✓ Plan ensures all RAG responses will be grounded in the ingested content from my-web docs
✓ Will implement verification mechanisms to trace claims back to source documents

### Gate 2: Clarity for Academic Audience
✓ Backend-focused planning targets the specified audience of backend developers
✓ Technical terminology will be precise and well-defined

### Gate 3: Reproducibility
✓ Implementation plan will include traceable steps from the book content
✓ Code examples will be reproducible with provided credentials

### Gate 4: Rigor
✓ Will prioritize peer-reviewed sources from ingested content
✓ Will flag non-peer-reviewed elements clearly

### Gate 5: Traceable Factual Claims
✓ All claims will be linked to specific sections in the ingested content
✓ Will use proper citation format in implementation

### Gate 6: Zero Plagiarism
✓ Original implementation code will be generated
✓ Will avoid copying text verbatim from sources

### Gate 7: Security & Credential Handling
✓ Plan incorporates secure credential handling using environment variables
✓ Will not expose credentials in production code

## Project Structure

### Documentation (this feature)

```text
specs/001-rag-chatbot-spec/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
│   └── chatbot-api.yaml # OpenAPI specification
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── main.py          # FastAPI application entry point
│   ├── models/          # Pydantic models and data structures
│   │   ├── chat.py      # Chat-related models
│   │   ├── document.py  # Document models
│   │   └── user.py      # User models
│   ├── services/        # Business logic services
│   │   ├── chat_service.py       # Chat completion service
│   │   ├── document_service.py   # Document management service
│   │   ├── vector_store.py       # Qdrant integration
│   │   └── llm_service.py        # LLM interaction service
│   ├── database/        # Database operations
│   │   ├── models.py    # SQLAlchemy models
│   │   ├── session.py   # Database session management
│   │   └── init.py      # Database initialization
│   ├── ingestion/       # Document ingestion pipeline
│   │   ├── batch_ingest.py   # Batch ingestion script
│   │   └── chunker.py        # Document chunking logic
│   └── api/             # API routes
│       ├── v1/          # Version 1 API routes
│       │   ├── chat.py      # Chat endpoints
│       │   ├── documents.py # Document endpoints
│       │   └── health.py    # Health check endpoint
│       └── deps.py      # Dependency injection
├── tests/               # Test suite
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── contract/        # API contract tests
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment variables
├── .gitignore          # Git ignore rules
└── README.md           # Project documentation
```

**Structure Decision**: Backend service structure selected for RAG chatbot implementation. The system follows a service-oriented architecture with clear separation between API layer, business logic, data models, and external service integrations (Qdrant, Neon Postgres, OpenRouter).

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |

## Outcome

- ✅ Impact: Created comprehensive implementation plan for RAG chatbot system with all required components
- 🧪 Tests: No tests created in planning phase
- 📁 Files: Created/updated plan.md, research.md, data-model.md, quickstart.md, and chatbot-api.yaml
- 🔁 Next prompts: Implementation phase with /sp.tasks and /sp.implement
- 🧠 Reflection: Plan addresses all requirements from user input including credential handling, fallback mechanisms, and security considerations

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Not applicable
- Prompt variant (if applicable): Not applicable
- Next experiment (smallest change to try): Begin implementation with task generation