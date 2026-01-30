# 🔍 Retrieval-Augmented Generation (RAG) System from Scratch

An end-to-end **Retrieval-Augmented Generation (RAG)** system built from first principles, demonstrating how modern AI applications combine **semantic search** with **large language models** to produce **grounded, reliable answers with citations**.

This project intentionally avoids high-level frameworks at first and implements the core mechanics manually to build **deep system-level understanding** of real-world LLM systems.

---

## 🚀 What This Project Does

Given a natural-language query, the system:

1. **Retrieves** the most relevant document chunks using dense vector similarity (FAISS)
2. **Augments** the user prompt with retrieved context
3. **Generates** a grounded answer using an LLM
4. **Validates** the output against a strict JSON schema
5. **Refuses** to answer when information is missing
6. **Cites** the exact document chunks used

> This is **real RAG** — not just semantic search and not just generation.

---

## 🧠 Why This Project Matters

Modern LLM applications fail not because models are weak, but because:

- Context is poorly retrieved  
- Hallucinations go unchecked  
- Outputs aren’t validated  
- Cost and failure modes are ignored  

This project focuses on **engineering reliability**, not model hype.

---

## 🏗️ System Architecture

# 🔍 Retrieval-Augmented Generation (RAG) System from Scratch

An end-to-end **Retrieval-Augmented Generation (RAG)** system built from first principles, demonstrating how modern AI applications combine **semantic search** with **large language models** to produce **grounded, reliable answers with citations**.

This project intentionally avoids high-level frameworks at first and implements the core mechanics manually to build **deep system-level understanding** of real-world LLM systems.

---

## 🚀 What This Project Does

Given a natural-language query, the system:

1. **Retrieves** the most relevant document chunks using dense vector similarity (FAISS)
2. **Augments** the user prompt with retrieved context
3. **Generates** a grounded answer using an LLM
4. **Validates** the output against a strict JSON schema
5. **Refuses** to answer when information is missing
6. **Cites** the exact document chunks used

> This is **real RAG** — not just semantic search and not just generation.

---

## 🧠 Why This Project Matters

Modern LLM applications fail not because models are weak, but because:

- Context is poorly retrieved  
- Hallucinations go unchecked  
- Outputs aren’t validated  
- Cost and failure modes are ignored  

This project focuses on **engineering reliability**, not model hype.

---

## 🏗️ System Architecture

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


---

## 📁 Project Structure

semantic-search-Engine/
│
├── data/ # Raw text documents
│
├── chunking/
│ ├── fixed.py
│ ├── overlap.py
│ └── recursive.py
│
├── embeddings/
│ └── embedder.py # Sentence-transformer embedder
│
├── index/
│ ├── build_index.py # Build FAISS index + metadata
│ └── search.py # FAISS retrieval logic
│
├── rag/
│ ├── prompt_builder.py # RAG prompt + guardrails
│ ├── rag_pipeline.py # Retrieval + generation
│ └── test_rag.py # End-to-end demo
│
├── evaluation/
│ └── gold_queries.json # Gold set for evaluation
│
├── index/
│ ├── chunk_index.faiss
│ └── chunk_metadata.json
│
├── .env # OPENAI_API_KEY (gitignored)
├── requirements.txt
└── README.md


---

## 🔑 Key Concepts Implemented

### 1️⃣ Semantic Search (No LLM Involved)
- Sentence-level embeddings  
- Vector normalization  
- Cosine similarity  
- FAISS indexing  
- Top-K nearest-neighbor retrieval  

---

### 2️⃣ Chunking Strategies
- Fixed-size chunking  
- Overlapping windows  
- Structure-aware (recursive) chunking  
- Tradeoffs between **recall, precision, and cost**

---

### 3️⃣ Retrieval-Augmented Generation (RAG)
- Context injection into prompts  
- Explicit grounding rules  
- Source citation tracking  
- Refusal behavior when context is insufficient  

---

### 4️⃣ Prompt Engineering (Engineering-Grade)
- System-level instruction dominance  
- Explicit **“don’t guess”** rules  
- Guardrails against hallucination  
- Context-only answering  
- Confidence calibration  

---

### 5️⃣ Output Validation
- Strict JSON schema  
- Automatic retries on invalid output  
- Fail-fast behavior after repeated violations  

---

## 🧪 Example Queries & Behavior

### ✅ Grounded Answer

**Query:** *What is a vector database?*

```json
{
  "answer": "A vector database is a data store specialized for handling data represented as high-dimensional vectors, enabling efficient similarity search over embeddings.",
  "confidence": 1.0,
  "used_sources": [
    "vector_databases.txt chunk=1",
    "vector_databases.txt chunk=4"
  ]
}


🚫 Refusal (Correct Behavior)

Query: Who won the 2035 Cricket World Cup?

{
  "answer": "I don't know",
  "confidence": 0.3,
  "used_sources": []
}
```

This refusal is intentional and correct — no hallucination.

📊 Evaluation Results

The semantic retrieval layer was evaluated using a gold query set, measuring document hit@3.

Doc Hit@3: 5 / 10 = 50%

Key Insight

Abstract conceptual queries often require LLM-based synthesis beyond pure embedding retrieval.

This motivates:

Query rewriting

Improved chunking

Hybrid retrieval strategies

🧠 Engineering Lessons Learned

Embeddings ≠ answers — retrieval quality dominates RAG performance

Chunking strategy often matters more than model choice

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

2️⃣ Run the RAG demo
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

Systems thinking


## FLOW OF THE PROJECT:


The full flow of your project (end-to-end, very detailed)

I’ll describe two flows:

A) Offline flow (indexing) — done once per corpus update
B) Online flow (query answering) — per user query
A) OFFLINE FLOW — Build the index

You run:

python -m index.build_index

Step A1 — Load documents from data/

reads each .txt

stores:

doc_id = filename

text = file content

Step A2 — Chunk each document

For each doc:

apply chosen chunker (fixed/overlap/recursive)

output many chunks:

each chunk has:

doc_id

chunk_id

text

Step A3 — Embed each chunk (one-time)

call your Embedder.embed(chunks_texts)

output an array:

shape: (num_chunks, embedding_dim)

Step A4 — Normalize embeddings (if used)

for cosine similarity you typically normalize vectors to unit length

you already do this in your Embedder

Step A5 — Build FAISS index

create FAISS index (usually inner product for normalized vectors)

add all chunk vectors to index

Step A6 — Save to disk

save index:

index/chunk_index.faiss

save metadata mapping:

index/chunk_metadata.json

maps row i → {doc_id, chunk_id, text}

✅ Offline done.

B) ONLINE FLOW — Answering a user query (RAG)

User asks: “What is FAISS used for?”

You run:

python -m rag.test_rag


or later your API will do it.

Step B1 — Guardrails pre-check (optional stage)

detect prompt injection

detect unsafe requests

set policy: “context-only answering”

Step B2 — Query embedding

embed the user query using same embedder

vector: shape (1, dim)

normalized

Step B3 — Retrieve top-k chunks from FAISS

index.search(q_vec, top_k)

returns:

indices: chunk ids in FAISS

scores: similarity scores

map indices → metadata rows

produce retrieved chunks:

list of {doc_id, chunk_id, text}

Step B4 — Build the RAG prompt

You construct messages:

System message:

strict rules

output JSON schema

“answer only from context”

“refuse if missing”

User message includes:

the retrieved CONTEXT chunks (with IDs)

the QUESTION

Step B5 — Call the LLM (generation)

send messages to OpenAI

model outputs text (should be JSON)

Step B6 — Parse and validate output schema

parse JSON into AnswerSchema

if invalid:

retry 1–3 times

if still invalid:

raise runtime error

Step B7 — Post-check (faithfulness / refusal)

(Optional but you’ll implement)

verify citations refer to retrieved chunks

if not → reject or force refusal

Step B8 — Return answer to user

answer text

confidence

citations

✅ Online done.

Why this project is real engineering

Because you have:

reproducible offline pipeline

deterministic retrieval

explicit prompt contract

schema validation

refusal behavior

evaluation harness

That’s what real LLM teams build.

Failure modes

Production realism

It reflects how real LLM applications are built — not demos, but reliable systems.
