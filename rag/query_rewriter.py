# rag/query_rewriter.py
from __future__ import annotations
from typing import List

# Domain-aware expansions for FIA regulations
DRIVER_EXPANSIONS = [
    "driver shall",
    "driver must",
    "competitor or driver",
    "driver responsibilities",
    "driver penalties",
    "driver conduct",
]

SPRINT_EXPANSIONS = [
    "sprint session",
    "sprint qualifying",
    "sprint race",
    "sprint classification",
    "sprint grid",
]


def rewrite_query(query: str) -> List[str]:
    """
    Rewrite abstract user queries into concrete retrieval queries.
    Returns a list of queries to retrieve with.
    """
    q = query.lower()
    rewrites = [query]  # always keep original

    if "driver" in q:
        rewrites.extend(DRIVER_EXPANSIONS)

    if "sprint" in q:
        rewrites.extend(SPRINT_EXPANSIONS)

    # de-duplicate while preserving order
    seen = set()
    out = []
    for r in rewrites:
        if r not in seen:
            out.append(r)
            seen.add(r)

    return out
