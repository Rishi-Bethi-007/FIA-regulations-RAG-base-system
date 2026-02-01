# rag/test_rag.py
from __future__ import annotations

from rag.rag_pipeline import answer
from index.chroma_store import stats
from config import TOP_K


def _dedupe_for_display(hits: list[dict]) -> list[dict]:
    """
    De-dupe hits for printing so the CLI doesn't show repeats.
    (Does NOT change retrieval/rerank; display-only.)
    """
    seen = set()
    out = []
    for h in hits:
        m = h.get("meta", {})
        key = (m.get("source"), m.get("page"), m.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def run():
    print("\n🧠 FIA RAG CLI")
    print("Type a question and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    # Print collection stats once (not every loop)
    s = stats()
    print(f"📦 Chroma collection: {s['name']} | vectors={s['count']}")
    print(f"🔎 Showing top {TOP_K} hits in debug output.\n")

    while True:
        q = input("Ask a question: ").strip()

        if not q:
            continue

        if q.lower() in {"exit", "quit"}:
            print("\n👋 Exiting RAG CLI.")
            break

        text, answer_hits = answer(q)

        print("\n--- ANSWER ---")
        print(text)

        # Display-only cleanup:
        # 1) dedupe repeated chunks
        # 2) show only top TOP_K hits
        display_hits = _dedupe_for_display(answer_hits)[:TOP_K]

        print("\n--- TOP HITS (from answer / reranked) ---")
        if not display_hits:
            print("(no hits)")
        else:
            for h in display_hits:
                m = h.get("meta", {})
                dist = h.get("distance", None)
                dist_s = f"{dist:.4f}" if isinstance(dist, (int, float)) else "NA"
                print(
                    f"- season={m.get('season')} "
                    f"{m.get('source')} p.{m.get('page')} "
                    f"dist={dist_s} "
                    f"rerank={h.get('rerank_score','MISSING')}"
                )

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    run()
