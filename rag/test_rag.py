from rag.rag_pipeline import run_rag

queries = [
    "What is a vector database?",
    "What is FAISS used for?",
    "What are transformers?",
    "What affects retrieval quality in RAG?",
    "Who won the 2035 Cricket World Cup?"
]

for q in queries:
    print("\n" + "=" * 80)
    print("QUERY:", q)
    try:
        result = run_rag(q)
        print(result.model_dump())
    except Exception as e:
        print("ERROR:", e)
