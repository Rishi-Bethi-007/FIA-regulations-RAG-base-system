import argparse
from email import parser
import json
from typing import List , Dict ,Any
from collections import defaultdict

import faiss
import numpy as np

from embeddings.embedder import Embedder

class FaissRetriever:
    """
    Thin wrapper around your existing load_index/load_metadata/search() functions
    so your RAG pipeline can just call: retriever.retrieve(query, top_k)
    """

    def __init__(self, index_path: str = "index/chunk_index.faiss", meta_path: str = "index/chunk_metadata.json"):
        self.index = load_index(index_path)
        self.metadata = load_metadata(meta_path)

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Index vectors ({self.index.ntotal}) != metadata rows ({len(self.metadata)}). "
                "Your mapping is inconsistent."
            )

        self.embedder = Embedder()

    def retrieve(self, query: str, top_k: int = 5):
        results = search(
            query=query,
            embedder=self.embedder,
            index=self.index,
            metadata=self.metadata,
            top_k=top_k
        )

        # Convert to the exact shape your prompt_builder expects
        return [
            {
                "doc_id": r["doc_id"],
                "chunk_id": r["chunk_id"],
                "text": r["text"]
            }
            for r in results
        ]





def load_index(index_path: str) -> faiss.Index:
    """Load a FAISS index from the specified path."""
    return faiss.read_index(index_path)

def load_metadata(meta_path: str) -> List[Dict[str, Any]]:
    """Load metadata from a JSON file."""
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def search(
        query:str,
        embedder: Embedder,
        index: faiss.Index,
        metadata: List[Dict[str, Any]],
        top_k: int = 5,
) -> List[Dict[str,Any]]:
    # 1) Embed + normalize query (Embedder already normalizes)
    q_vec = embedder.embed([query]).numpy().astype('float32') # (1, dim) 

    # 2) Search index for top-k most similar vectors
    scores, indices = index.search(q_vec, top_k)  # (1, top_k) each

    # 3) Map indices back to chunk text + metadata and return results 
    results = []
    for rank, (score,idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue  # Skip invalid indices

        item = metadata[idx]
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "doc_id": item["doc_id"],
                "chunk_id": item["chunk_id"],
                "text": item["text"],
            }
        )
    return results

def rank_documents(results):
    """
    Aggregate chunk-level results into document-level ranking.
    """
    doc_scores = defaultdict(list)

    for r in results:
        doc_scores[r["doc_id"]].append(r["score"])
    
    ranked_docs =[]
    for doc_id, scores in doc_scores.items():
        ranked_docs.append({
            "doc_id": doc_id,
            "score": max(scores), # max-pooling
            "num_chunks": len(scores),
        })

    ranked_docs.sort(key=lambda x: x["score"], reverse=True)
    return ranked_docs


def main():
     parser = argparse.ArgumentParser(description="Semantic search over FAISS index.")
     parser.add_argument("query", type=str, help="Search query text in quotes")
     parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
     parser.add_argument("--index_path", type=str, default="index/chunk_index.faiss")
     parser.add_argument("--meta_path", type=str, default="index/chunk_metadata.json")

     args = parser.parse_args()

     print("Loading faiss index...")
     index = load_index(args.index_path)

     print('Loading metadata...')
     metadata = load_metadata(args.meta_path)

     if index.ntotal != len(metadata):
         raise ValueError( f"Index vectors ({index.ntotal}) != metadata rows ({len(metadata)}). "
            "Your mapping is inconsistent.")
     
     print("Initializing embedder...")
     embedder = Embedder()

     print("\nQuery:", args.query)
     results = search(
         query=args.query,
            embedder=embedder,
            index=index,
            metadata=metadata,
            top_k=args.top_k,
     )

     print("\n==== Results ====")
     for r in results:
         print(f"\n Rank: {r['rank']} | Score: {r['score']:.4f} | Doc ID: {r['doc_id']} | Chunk ID: {r['chunk_id']}\nText: {r['text']}")

     doc_ranks = rank_documents(results)

     print("\n=== DOCUMENT RANKING ===")
     for i, d in enumerate(doc_ranks, start=1):
        print(
            f"Doc Rank {i} | Score {d['score']:.4f} "
            f"| Doc ID: {d['doc_id']} | Chunks matched: {d['num_chunks']}"
        )


if __name__ == "__main__":
    main()