🔍 Retrieval-Augmented Generation (RAG) System from Scratch

An end-to-end Retrieval-Augmented Generation (RAG) system built from first principles, demonstrating how modern AI applications combine semantic search with large language models to produce grounded, reliable answers with source citations.

This project intentionally avoids high-level frameworks at first and implements the core mechanics manually to build deep system-level understanding.

🚀 What This Project Does

Given a natural-language query, the system:

Retrieves the most relevant document chunks using dense vector similarity (FAISS)

Augments the user prompt with retrieved context

Generates a grounded answer using an LLM

Validates the output against a strict JSON schema

Refuses to answer when information is missing

Cites the exact document chunks used

This is real RAG, not just semantic search and not just generation.

🧠 Why This Project Matters

Modern LLM applications fail not because models are weak, but because:

context is poorly retrieved

hallucinations go unchecked

outputs aren’t validated

costs and failure modes are ignored

This project focuses on engineering reliability, not model hype.

🏗️ System Architecture
Documents
   ↓
Chunking
   ↓
Embeddings (Sentence Transformers)
   ↓
FAISS Vector Index
   ↓
Query Embedding
   ↓
Top-K Retrieval
   ↓
Prompt Assembly (with guardrails)
   ↓
LLM Generation (OpenAI)
   ↓
Schema Validation + Refusal Logic


📁 Project Structure
semantic-search-Engine/
│
├── data/                  # Raw text documents
│
├── chunking/
│   ├── fixed.py
│   ├── overlap.py
│   └── recursive.py
│
├── embeddings/
│   └── embedder.py        # Sentence-transformer embedder
│
├── index/
│   ├── build_index.py     # Build FAISS index + metadata
│   └── search.py          # FAISS retrieval logic
│
├── rag/
│   ├── prompt_builder.py  # RAG prompt + guardrails
│   ├── rag_pipeline.py    # Retrieval + generation
│   └── test_rag.py        # End-to-end demo
│
├── evaluation/
│   └── gold_queries.json  # Gold set for evaluation
│
├── index/
│   ├── chunk_index.faiss
│   └── chunk_metadata.json
│
├── .env                   # OPENAI_API_KEY (gitignored)
├── requirements.txt
└── README.md


🔑 Key Concepts Implemented
1️⃣ Semantic Search (No LLM Involved)

Sentence-level embeddings

Vector normalization

Cosine similarity

FAISS indexing

Top-K nearest-neighbor retrieval

2️⃣ Chunking Strategies

Fixed-size chunking

Overlapping windows

Structure-aware (recursive) chunking

Tradeoffs between recall, precision, and cost

3️⃣ Retrieval-Augmented Generation

Context injection into prompts

Explicit grounding rules

Source citation tracking

Refusal behavior when context is insufficient

4️⃣ Prompt Engineering (Engineering-Grade)

System-level instruction dominance

Explicit “don’t guess” rules

Guardrails against hallucination

Context-only answering

Confidence calibration

5️⃣ Output Validation

Strict JSON schema

Automatic retries on invalid output

Fail-fast behavior after repeated violations

🧪 Example Queries & Behavior
Grounded Answer

Query: What is a vector database?

{
  "answer": "A vector database is a data store specialized for handling data represented as high-dimensional vectors, enabling efficient similarity search over embeddings.",
  "confidence": 1.0,
  "used_sources": [
    "vector_databases.txt chunk=1",
    "vector_databases.txt chunk=4"
  ]
}

Refusal (Correct Behavior)

Query: Who won the 2035 Cricket World Cup?

{
  "answer": "I don't know",
  "confidence": 0.3,
  "used_sources": []
}


This refusal is intentional and correct — no hallucination.

📊 Evaluation Results

I evaluated the semantic retrieval layer using a gold query set and measured document hit@3.

Result:

Doc Hit@3: 5/10 = 50%

Key Insight

Abstract conceptual queries often require LLM-based synthesis beyond pure embedding retrieval.

This directly motivates:

query rewriting

better chunking

hybrid retrieval strategies (future work)

🧠 Engineering Lessons Learned

Embeddings ≠ answers — retrieval quality dominates RAG performance

Chunking strategy affects accuracy more than model choice

Guardrails matter more than clever prompts

LLMs must be treated as unreliable collaborators

Environment setup is a real engineering challenge (Conda isolation used)

🛠️ Tech Stack

Python 3.10

Sentence Transformers

FAISS (CPU)

OpenAI API

Pydantic

Conda (isolated environment)

⚙️ How to Run
1️⃣ Build the index (one-time)
python -m index.build_index

2️⃣ Run RAG demo
python -m rag.test_rag

🔒 Security & Best Practices

API keys stored in .env

.env is gitignored

No secrets committed

Deterministic failure handling

🧭 Roadmap / Next Improvements

Query rewriting for better retrieval

RAG-specific evaluation metrics (faithfulness, groundedness)

Confidence calibration

Hybrid lexical + dense retrieval

Framework comparison (LangChain / LlamaIndex)

Agentic retrieval planning

🧑‍💻 Author Notes

This project was built as part of a deliberate transition from ML theory → AI engineering, focusing on:

systems thinking

failure modes

production realism

It reflects how real LLM applications are built — not demos, but reliable systems.
