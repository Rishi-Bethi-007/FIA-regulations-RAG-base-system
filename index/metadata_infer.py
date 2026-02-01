# index/metadata_infer.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any

# Regex helpers
_RE_YEAR = re.compile(r"(20\d{2})")
_RE_ISSUE = re.compile(r"\bissue[\s_-]*(\d{1,2})\b", re.IGNORECASE)
_RE_PUBLISHED_ISO = re.compile(r"\b(20\d{2})[-_/](\d{1,2})[-_/](\d{1,2})\b")


def infer_metadata(pdf_path: Path, dataset_name: str = "fia") -> Dict[str, Any]:
    """
    Infer doc-level metadata from filename + folder structure.

    Designed to scale: no per-doc manual entries required.

    Returns fields that are stable and queryable:
      - dataset, doc_type
      - season (year)
      - regulation_type (sporting/technical)
      - issue (optional)
      - published (optional)
      - tenant (default)
      - allowed_roles (optional via overrides, not inferred)
    """
    name = pdf_path.name
    low = name.lower()
    parts = [p.lower() for p in pdf_path.parts]

    meta: Dict[str, Any] = {}

    # Dataset identity
    meta["dataset"] = dataset_name
    meta["doc_type"] = "fia_f1_regulations"  # keep stable across sporting/technical

    # ✅ Season = MAX year in filename (season year beats published year)
    years = [int(y) for y in _RE_YEAR.findall(name)]
    if years:
        meta["season"] = max(years)



    # Regulation type
    if "sporting" in low or "sporting" in parts:
        meta["regulation_type"] = "sporting"
    elif "technical" in low or "technical" in parts:
        meta["regulation_type"] = "technical"
    elif "operational" in low or "operational" in parts:
        meta["regulation_type"] = "operational"

    # Issue number (if present in filename)
    m = _RE_ISSUE.search(name)
    if m:
        meta["issue"] = int(m.group(1))

    # Published date (if present like 2025-02-26 in filename)
    m = _RE_PUBLISHED_ISO.search(name)
    if m:
        yyyy, mm, dd = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
        meta["published"] = f"{yyyy}-{mm}-{dd}"

    # Tenant default (can be overridden)
    meta["tenant"] = "fia"

    return meta
