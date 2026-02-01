# FIA Regulations RAG System (Base v1)

An industry-grade Retrieval-Augmented Generation (RAG) system for querying
Formula 1 regulations across multiple seasons (2018–2026).

This project focuses on **correct system design**, not frameworks.

## Features

- Sentence-aware chunking
- Header/footer removal for regulatory PDFs
- Automatic metadata inference (year, regulation type, issue)
- Vector search with Chroma
- Comparison-aware retrieval (per-season recall)
- Wide recall + reranking
- Faithful, citation-grounded answers
- CLI interface for interactive querying

## Example Queries

- Compare sprint rules in 2021 vs 2026
- What are the sporting regulation changes in 2025?
- What does the FIA say about safety car restarts?

## Architecture

PDFs → Clean → Chunk → Embed → Vector DB  
Query → Rewrite → Recall (wide) → Rerank → Generate (grounded)

## Tech Stack

- Python
- OpenAI embeddings + generation
- ChromaDB
- pypdf
- Regex-based metadata inference

## Why this project

Most RAG demos ignore:
- retrieval bias
- comparison queries
- noisy PDFs
- evaluation
- scalability concerns

This project addresses those explicitly.

## Status

**Frozen base version (v1).**

Future work (separate repos):
- Pinecone-backed RAG
- LangChain / LlamaIndex comparison
- Agent workflows
- Async ingestion
