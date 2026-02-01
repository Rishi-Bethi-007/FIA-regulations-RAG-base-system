# index/search.py
from config import TOP_K
from embeddings.embedder import embed_query
from index.chroma_store import query as chroma_query

def search(query_text: str, k: int = TOP_K, where: dict | None = None):
    q_emb = embed_query(query_text)
    return chroma_query(q_emb, k=k, where=where)

#if __name__ == "__main__":
    # Example: unfiltered
 #   hits = search("What is attention mechanism?")
  #  for h in hits:
   #     print(h["meta"], h["distance"])

    # Example: filter to a specific source PDF (META Data Filtering)
    #hits = search("What is attention mechanism?", where={"source": "my_doc.pdf"})
