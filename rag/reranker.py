# rag/reranker.py
from openai import OpenAI
from config import OPENAI_API_KEY, RERANK_MODEL, RERANK_MAX_CHARS

client = OpenAI(api_key=OPENAI_API_KEY)

def _clip(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"

def rerank(query: str, hits: list[dict], top_k: int) -> list[dict]:
    """
    LLM reranker:
    - Input: query + candidate chunks (hits)
    - Output: same hits but filtered/sorted by relevance

    Each hit must have:
      hit["text"], hit["meta"], hit["distance"]
    """
    if not hits:
        return []

    # Build compact candidates list
    candidates = []
    for i, h in enumerate(hits, start=1):
        m = h.get("meta", {})
        candidates.append({
            "i": i,
            "season": m.get("season"),
            "source": m.get("source"),
            "page": m.get("page"),
            "text": _clip(h.get("text", ""), RERANK_MAX_CHARS),
        })

    prompt = f"""
You are a strict reranker for retrieval-augmented generation.

Task:
Given a QUESTION and a list of CANDIDATES, score each candidate for how useful it is
to answer the QUESTION using only that candidate's text.

Rules:
- Score range: 0 to 100 (100 = directly answers the question with specific rules/details).
- Prefer candidates that contain explicit rules/procedures/definitions related to the question.
- Penalize boilerplate, definitions unrelated to the question, or generic mentions.
- You MUST return valid JSON only, no extra text.

Return JSON schema:
{{"ranking":[{{"i": <int>, "score": <int>}} ...]}}

QUESTION:
{query}

CANDIDATES:
{candidates}
""".strip()

    resp = client.responses.create(
        model=RERANK_MODEL,
        input=prompt,
    )

    # Parse JSON safely (simple approach)
    import json
    try:
        data = json.loads(resp.output_text)
        ranking = data.get("ranking", [])
    except Exception:
    # Fallback: keep original order, but still attach scores (0)
        for h in hits:
            h["rerank_score"] = 0
        return hits[:top_k]

    # Build score map
    score_by_i = {}
    for r in ranking:
        try:
            i = int(r["i"])
            score = int(r["score"])
            score_by_i[i] = score
        except Exception:
            continue

    # Attach scores and sort
    scored = []
    for idx, h in enumerate(hits, start=1):
        score = score_by_i.get(idx, 0)
        hh = dict(h)
        hh["rerank_score"] = score
        scored.append(hh)

    scored.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return scored[:top_k]
