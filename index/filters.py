# index/filters.py
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Expandable range (you said 2018–2026)
MIN_SEASON = 2018
MAX_SEASON = 2026

# Unified doc_type used by inference (recommended)
DOC_TYPE = "fia_f1_regulations"


@dataclass
class QueryPlan:
    is_comparison: bool
    seasons: List[int]           # seasons explicitly mentioned (or defaulted)
    where: Dict[str, Any]        # Chroma filter: doc_type + optional constraints


def _extract_years(query_text: str) -> List[int]:
    """
    Finds years like 2018..2026 in the query.
    Returns sorted unique years within allowed range.
    """
    # Find any 4-digit year starting with 20xx
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", query_text)]
    years = [y for y in years if MIN_SEASON <= y <= MAX_SEASON]
    return sorted(set(years))


def _is_comparison_query(query_text: str) -> bool:
    """
    Heuristics to detect user intent: comparing across seasons.
    """
    q = query_text.lower()
    return (
        "compare" in q
        or " vs " in q
        or "versus" in q
        or "difference" in q
        or "different" in q
        or "changed" in q
        or "change" in q
        or "between" in q
    )


def _detect_regulation_type(query_text: str) -> Optional[str]:
    """
    Detect whether user explicitly wants sporting or technical regs.
    Returns: "sporting", "technical", or None (not specified).
    """
    q = query_text.lower()

    # If both are mentioned, return None (ambiguous) and let retrieval handle it broadly.
    wants_sporting = "sporting" in q
    wants_technical = "technical" in q

    if wants_sporting and not wants_technical:
        return "sporting"
    if wants_technical and not wants_sporting:
        return "technical"
    return None


def build_plan(query_text: str) -> QueryPlan:
    """
    Builds a plan used by the RAG pipeline:
    - is_comparison: whether to do balanced retrieval across seasons
    - seasons: target seasons (if explicitly mentioned or defaulted)
    - where: Chroma filter dict
    """
    years = _extract_years(query_text)
    is_comp = _is_comparison_query(query_text)

    # Base filter: always restrict to FIA dataset doc_type
    base: Dict[str, Any] = {"doc_type": DOC_TYPE}

    # Optional: restrict to sporting/technical if user asked
    reg_type = _detect_regulation_type(query_text)
    if reg_type:
        base = {"$and": [base, {"regulation_type": reg_type}]}

    # Case 1: exactly one year → single-season query
    if len(years) == 1:
        return QueryPlan(
            is_comparison=False,
            seasons=years,
            where={"$and": [base, {"season": years[0]}]},
        )

    # Case 2: multiple years → comparison query
    if len(years) >= 2:
        years = sorted(set(years))
        return QueryPlan(
            is_comparison=True,
            seasons=years,
            where={"$and": [base, {"season": {"$in": years}}]},
        )

    # Case 3: no year specified
    # If it sounds like comparison, default to latest two seasons
    if is_comp:
        default_years = [MAX_SEASON - 1, MAX_SEASON]  # e.g. 2025, 2026
        return QueryPlan(
            is_comparison=True,
            seasons=default_years,
            where={"$and": [base, {"season": {"$in": default_years}}]},
        )

    # Case 4: broad question across all seasons (no season filter)
    return QueryPlan(
        is_comparison=False,
        seasons=[],
        where=base,
    )
