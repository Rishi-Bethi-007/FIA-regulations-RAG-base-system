# rag/rag_pipeline.py
from __future__ import annotations

from typing import List, Dict, Any
from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    GEN_MODEL,
    TOP_K,
    RERANK_ENABLED,
    RECALL_K,
)

from index.search import search
from index.filters import build_plan


from rag.query_rewriter import rewrite_query
from rag.reranker import rerank


client = OpenAI(api_key=OPENAI_API_KEY)


# -----------------------------
# Formatting helpers
# -----------------------------
def format_citations(hits: List[Dict]) -> str:
    """Simple citation format: [i] source p.X"""
    cites = []
    for i, h in enumerate(hits, start=1):
        m = h["meta"]
        cites.append(f"[{i}] {m.get('source')} p.{m.get('page')}")
    return "\n".join(cites)


def build_context(hits: List[Dict]) -> str:
    """Keep chunk text + inline metadata for traceability."""
    blocks = []
    for i, h in enumerate(hits, start=1):
        m = h["meta"]
        blocks.append(
            f"CHUNK {i} | source={m.get('source')} | page={m.get('page')} | chunk={m.get('chunk_index')}\n"
            f"{h.get('text', '')}"
        )
    return "\n\n---\n\n".join(blocks)


# -----------------------------
# Comparison guard
# -----------------------------
def enforce_per_season_min(
    hits: List[Dict],
    seasons: List[int],
    top_k: int,
    min_per_season: int = 2,
) -> List[Dict]:
    """
    Ensure final evidence contains at least `min_per_season`
    chunks from each season when doing comparisons.
    """
    picked: List[Dict] = []
    used = set()

    def _key(h: Dict):
        m = h["meta"]
        return (m.get("source"), m.get("page"), m.get("chunk_index"))

    # 1) Guarantee minimum per season
    for season in seasons:
        season_hits = [h for h in hits if h["meta"].get("season") == season]
        for h in season_hits[:min_per_season]:
            k = _key(h)
            if k not in used:
                picked.append(h)
                used.add(k)

    # 2) Fill remaining slots by best overall
    for h in hits:
        if len(picked) >= top_k:
            break
        k = _key(h)
        if k not in used:
            picked.append(h)
            used.add(k)

    return picked[:top_k]


# -----------------------------
# Main entrypoint
# -----------------------------
def answer(query: str, where: Dict[str, Any] | None = None):

    # -----------------------------
    # Query plan
    # -----------------------------
    plan = build_plan(query) if where is None else None

    # -----------------------------
    # Query rewriting
    # -----------------------------
    queries = rewrite_query(query)
    hits: List[Dict] = []

    if plan and plan.is_comparison and plan.seasons:
        recall_per_season = max(10, RECALL_K // len(plan.seasons))

        for season in plan.seasons:
            season_where = {
                "$and": [
                    {"doc_type": "fia_f1_regulations"},
                    {"season": season},
                ]
            }

            for qx in queries:
                hits.extend(search(qx, k=recall_per_season, where=season_where))

    else:
        base_where = where if where is not None else (plan.where if plan else None)

        for qx in queries:
            hits.extend(search(qx, k=RECALL_K, where=base_where))

        # -----------------------------
        # 2) RERANK (PRECISION)
        # -----------------------------
    if RERANK_ENABLED:
        hits = rerank(query, hits, top_k=max(TOP_K, 12))
    else:
        hits = hits[:max(TOP_K, 12)]

        # -----------------------------
        # 2.5) COMPARISON BALANCING
        # -----------------------------
        if plan and plan.is_comparison and plan.seasons:
            hits = enforce_per_season_min(
                hits=hits,
                seasons=plan.seasons,
                top_k=TOP_K,
                min_per_season=2,
            )
        else:
            hits = hits[:TOP_K]

    # -----------------------------
    # 3) GENERATION
    # -----------------------------
    context = build_context(hits)
    citations = format_citations(hits)

    prompt = f"""
You are a RAG assistant.
Rules:
- Answer ONLY using the provided CONTEXT.
- If the CONTEXT does not contain the answer, say: "I don't know based on the provided documents."
- Include citations in the form [1], [2], ... corresponding to the chunks.
- Do NOT repeat the same rule or sentence more than once.
- Group similar rules together and summarize them once.
If the question asks for a comparison:
    - Clearly separate rules for each season.
    - Then list explicit differences.
    - If no difference is stated in the documents, say so explicitly.

CONTEXT:
{context}

QUESTION:
{query}

CITATION INDEX:
{citations}
""".strip()

    resp = client.responses.create(
        model=GEN_MODEL,
        input=prompt,
    )

    return resp.output_text, hits
