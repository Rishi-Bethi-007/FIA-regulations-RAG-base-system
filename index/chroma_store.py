# index/chroma_store.py
import chromadb
from chromadb.config import Settings
from config import CHROMA_DIR, CHROMA_COLLECTION

_client = None
_collection = None

def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client

def get_collection():
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(name=CHROMA_COLLECTION)
        # Helpful when A/B testing:
        print(f"📦 Using Chroma collection: {CHROMA_COLLECTION} (dir={CHROMA_DIR})")
    return _collection

def upsert_chunks(
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
):
    if not (len(ids) == len(documents) == len(embeddings) == len(metadatas)):
        raise ValueError(
            f"Length mismatch: ids={len(ids)} docs={len(documents)} "
            f"embeddings={len(embeddings)} metas={len(metadatas)}"
        )

    col = get_collection()
    col.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

def query(
    query_embedding: list[float],
    k: int,
    where: dict | None = None,
):
    col = get_collection()
    res = col.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where or None,
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    out = []
    for doc, meta, dist in zip(docs, metas, dists):
        out.append({"text": doc, "meta": meta, "distance": dist})
    return out

def stats():
    col = get_collection()
    return {"name": col.name, "count": col.count(), "dir": CHROMA_DIR}
