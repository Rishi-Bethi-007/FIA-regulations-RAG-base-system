from embeddings.embedder import Embedder
from index.search import FaissRetriever 

class Retriever:
    def __init__(self, index_path: str, metadata_path: str):
        self.embedder = Embedder()
        self.index = FaissRetriever.load(index_path, metadata_path)

    def retrieve(self, query: str, top_k: int = 5):
        """
        Returns a list of chunks:
        [{ doc_id, chunk_id, text }, ...]
        """
        query_embedding = self.embedder.embed_query(query)
        results = self.index.search(query_embedding, top_k=top_k)

        chunks = []
        for r in results:
            chunks.append({
                "doc_id": r["doc_id"],
                "chunk_id": r["chunk_id"],
                "text": r["text"],
            })

        return chunks
