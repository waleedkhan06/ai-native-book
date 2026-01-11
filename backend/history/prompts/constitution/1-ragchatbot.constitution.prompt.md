---
id: 1
title: RAG Chatbot Constitution
stage: constitution
date: 2026-01-11
surface: agent
model: Claude Sonnet 4.5
feature: none
branch: main
user: claude-code
command: sp.constitution
labels: [constitution, rag, chatbot]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
  - none
---

## Prompt

RAG Chatbot System Prompt (sp.constitution)
You are a specialized RAG assistant for the research paper titled "Research Paper on AI-Native Software Development." Your responses must be grounded exclusively in the ingested content from the my-web docs folder (indexed via Qdrant and Neon). Adhere strictly to this constitution, which enforces the project's core principles, key standards, constraints, and success criteria.

[Full prompt content as provided by user]

## Response snapshot

Created RAG Chatbot constitution with core principles: Accuracy through Primary Source Verification, Clarity for Academic Audience, Reproducibility, Rigor, Traceable Factual Claims, Zero Plagiarism. Added standards, constraints, success criteria, and governance rules.

## Outcome

- ✅ Impact: Initial constitution created for RAG Chatbot system
- 🧪 Tests: none
- 📁 Files: .specify/memory/constitution.md
- 🔁 Next: Implement features guided by constitution