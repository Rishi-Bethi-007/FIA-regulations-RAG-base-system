from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Any, List

from index.pdf_loader import load_pdf_pages
from chunking.sentence_aware import chunk
from embeddings.embedder import embed_texts
from index.chroma_store import upsert_chunks
from index.metadata_infer import infer_metadata
from config import DATASET_NAME, CHUNK_SIZE, OVERLAP_SENTENCES


def stable_doc_id(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]


def build_index_from_pdfs(pdf_dir: str):
    pdf_dir_path = Path(pdf_dir)

    # 1) Load pages (pdf_loader handles cleaning)
    pages = load_pdf_pages(pdf_dir)

    all_ids: List[str] = []
    all_docs: List[str] = []
    all_metas: List[Dict[str, Any]] = []

    # Debug counters
    docs_seen = set()
    season_missing_sources = set()

    for p in pages:
        source = p["source"]  # filename only
        doc_id = stable_doc_id(source)

        # Only infer doc meta once per PDF (not per page)
        if source not in docs_seen:
            docs_seen.add(source)

        # 2) Infer metadata from filename
        pdf_path = pdf_dir_path / source
        doc_meta = infer_metadata(pdf_path, dataset_name=DATASET_NAME)

        # If inference fails, track it (should not happen after we fix infer_metadata)
        if "season" not in doc_meta:
            season_missing_sources.add(source)

        # 3) Chunk page text
        chunks = chunk(
            p["text"],
            chunk_size=CHUNK_SIZE,
            overlap_sentences=OVERLAP_SENTENCES,
        )

        # 4) Build per-chunk records
        for ci, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}-p{p['page']}-c{ci}"

            base_meta = {
                "doc_id": doc_id,
                "source": source,
                "page": p["page"],
                "chunk_index": ci,
                "chunker": "sentence",
                "chunk_size": CHUNK_SIZE,
                "overlap_sentences": OVERLAP_SENTENCES,
            }

            # ✅ merge inferred doc meta into chunk meta
            meta = {**doc_meta, **base_meta}

            all_ids.append(chunk_id)
            all_docs.append(chunk_text)
            all_metas.append(meta)

    print(f"📄 PDFs seen (from pages): {len(docs_seen)}")
    if season_missing_sources:
        print(f"⚠️ PDFs missing inferred season: {len(season_missing_sources)}")
        for s in list(season_missing_sources)[:20]:
            print(" -", s)

    # 5) Embed + upsert batches
    batch_size = 96
    for i in range(0, len(all_docs), batch_size):
        docs_b = all_docs[i:i + batch_size]
        ids_b = all_ids[i:i + batch_size]
        metas_b = all_metas[i:i + batch_size]

        embeds_b = embed_texts(docs_b)
        upsert_chunks(
            ids=ids_b,
            documents=docs_b,
            embeddings=embeds_b,
            metadatas=metas_b,
        )

    # Final proof prints
    print(
        f"✅ Indexed {len(all_docs)} chunks from {len(docs_seen)} PDFs "
        f"into collection (see CLI stats)."
    )


if __name__ == "__main__":
    build_index_from_pdfs("./data/pdfs")
