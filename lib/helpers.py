# lib/helpers.py
"""
Shared helpers for Université de Lorraine bibliometric dashboard.
Includes: taxonomy lookups, color palettes, blob parsers, utilities.

View-specific table builders should remain in their respective view files.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================================
# CONSTANTS
# ============================================================================

YEAR_START, YEAR_END = 2019, 2023
YEARS = list(range(YEAR_START, YEAR_END + 1))

# Domain order (by ID) - CORRECT ORDER
# Domain 1 = Life Sciences
# Domain 2 = Social Sciences
# Domain 3 = Physical Sciences
# Domain 4 = Health Sciences
DOMAIN_ORDER = [1, 2, 3, 4]
DOMAIN_NAMES_ORDERED = ["Life Sciences", "Social Sciences", "Physical Sciences", "Health Sciences"]

# Domain colors - mapped correctly by ID
DOMAIN_COLORS = {
    # By ID
    1: "#0CA750",   # Life Sciences (green)
    2: "#FFCB3A",   # Social Sciences (yellow)
    3: "#8190FF",   # Physical Sciences (blue)
    4: "#F85C32",   # Health Sciences (red/orange)
    # By name
    "Life Sciences": "#0CA750",
    "Social Sciences": "#FFCB3A",
    "Physical Sciences": "#8190FF",
    "Health Sciences": "#F85C32",
    # Fallback
    "Other": "#7f7f7f",
}

DOMAIN_EMOJI = {
    "Health Sciences": "🟥",
    "Life Sciences": "🟩",
    "Physical Sciences": "🟦",
    "Social Sciences": "🟨",
    "Other": "⬜",
}

# ============================================================================
# SAFE CONVERTERS
# ============================================================================

def safe_int(val: Any) -> int:
    """Convert value to int, return 0 on failure."""
    if pd.isna(val):
        return 0
    try:
        return int(float(str(val).strip().replace(",", "")))
    except (ValueError, TypeError):
        return 0


def safe_float(val: Any) -> float:
    """Convert value to float, return NaN on failure."""
    if pd.isna(val):
        return np.nan
    try:
        return float(str(val).strip().replace(",", "."))
    except (ValueError, TypeError):
        return np.nan


# ============================================================================
# TAXONOMY LOOKUPS
# ============================================================================
from lib.data_cache import get_topics_df
_TAXONOMY_CACHE: Dict[str, Any] = {}


def _ensure_taxonomy_loaded(topics_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ensure taxonomy is loaded. Pass topics_df or it will try to import from data_cache."""
    if "df" not in _TAXONOMY_CACHE:
        if topics_df is not None:
            _TAXONOMY_CACHE["df"] = topics_df
        else:
            _TAXONOMY_CACHE["df"] = get_topics_df()
    return _TAXONOMY_CACHE["df"]


def init_taxonomy(topics_df: pd.DataFrame) -> None:
    """Initialize taxonomy cache with provided DataFrame. Call once at app start."""
    _TAXONOMY_CACHE.clear()
    _TAXONOMY_CACHE["df"] = topics_df


def get_domain_id_to_name() -> Dict[int, str]:
    """Return {domain_id: domain_name} mapping."""
    if "domain_id_to_name" not in _TAXONOMY_CACHE:
        df = _ensure_taxonomy_loaded()
        _TAXONOMY_CACHE["domain_id_to_name"] = (
            df[["domain_id", "domain_name"]]
            .drop_duplicates()
            .set_index("domain_id")["domain_name"]
            .to_dict()
        )
    return _TAXONOMY_CACHE["domain_id_to_name"]


def get_domain_name_to_id() -> Dict[str, int]:
    """Return {domain_name: domain_id} mapping."""
    if "domain_name_to_id" not in _TAXONOMY_CACHE:
        _TAXONOMY_CACHE["domain_name_to_id"] = {v: k for k, v in get_domain_id_to_name().items()}
    return _TAXONOMY_CACHE["domain_name_to_id"]


def get_field_id_to_name() -> Dict[int, str]:
    """Return {field_id: field_name} mapping."""
    if "field_id_to_name" not in _TAXONOMY_CACHE:
        df = _ensure_taxonomy_loaded()
        _TAXONOMY_CACHE["field_id_to_name"] = (
            df[["field_id", "field_name"]]
            .drop_duplicates()
            .set_index("field_id")["field_name"]
            .to_dict()
        )
    return _TAXONOMY_CACHE["field_id_to_name"]


def get_field_name_to_id() -> Dict[str, int]:
    """Return {field_name: field_id} mapping."""
    if "field_name_to_id" not in _TAXONOMY_CACHE:
        _TAXONOMY_CACHE["field_name_to_id"] = {v: k for k, v in get_field_id_to_name().items()}
    return _TAXONOMY_CACHE["field_name_to_id"]


def get_field_id_to_domain_id() -> Dict[int, int]:
    """Return {field_id: domain_id} mapping."""
    if "field_id_to_domain_id" not in _TAXONOMY_CACHE:
        df = _ensure_taxonomy_loaded()
        _TAXONOMY_CACHE["field_id_to_domain_id"] = (
            df[["field_id", "domain_id"]]
            .drop_duplicates()
            .set_index("field_id")["domain_id"]
            .to_dict()
        )
    return _TAXONOMY_CACHE["field_id_to_domain_id"]


def get_subfield_id_to_name() -> Dict[int, str]:
    """Return {subfield_id: subfield_name} mapping."""
    if "subfield_id_to_name" not in _TAXONOMY_CACHE:
        df = _ensure_taxonomy_loaded()
        _TAXONOMY_CACHE["subfield_id_to_name"] = (
            df[["subfield_id", "subfield_name"]]
            .drop_duplicates()
            .set_index("subfield_id")["subfield_name"]
            .to_dict()
        )
    return _TAXONOMY_CACHE["subfield_id_to_name"]


def get_subfield_id_to_field_id() -> Dict[int, int]:
    """Return {subfield_id: field_id} mapping."""
    if "subfield_id_to_field_id" not in _TAXONOMY_CACHE:
        df = _ensure_taxonomy_loaded()
        _TAXONOMY_CACHE["subfield_id_to_field_id"] = (
            df[["subfield_id", "field_id"]]
            .drop_duplicates()
            .set_index("subfield_id")["field_id"]
            .to_dict()
        )
    return _TAXONOMY_CACHE["subfield_id_to_field_id"]


def get_subfield_id_to_domain_id() -> Dict[int, int]:
    """Return {subfield_id: domain_id} mapping."""
    if "subfield_id_to_domain_id" not in _TAXONOMY_CACHE:
        df = _ensure_taxonomy_loaded()
        _TAXONOMY_CACHE["subfield_id_to_domain_id"] = (
            df[["subfield_id", "domain_id"]]
            .drop_duplicates()
            .set_index("subfield_id")["domain_id"]
            .to_dict()
        )
    return _TAXONOMY_CACHE["subfield_id_to_domain_id"]


def get_field_order_by_domain() -> List[int]:
    """
    Return field IDs ordered by: domain order first (1,2,3,4), then field ID ascending within domain.
    """
    if "field_order_by_domain" not in _TAXONOMY_CACHE:
        df = _ensure_taxonomy_loaded()[["domain_id", "field_id"]].drop_duplicates()
        ordered = []
        for dom_id in DOMAIN_ORDER:
            fields = df.loc[df["domain_id"] == dom_id, "field_id"].tolist()
            ordered.extend(sorted(fields))
        _TAXONOMY_CACHE["field_order_by_domain"] = ordered
    return _TAXONOMY_CACHE["field_order_by_domain"]


def get_field_names_ordered() -> List[str]:
    """Return field names in domain-grouped order."""
    if "field_names_ordered" not in _TAXONOMY_CACHE:
        id2name = get_field_id_to_name()
        _TAXONOMY_CACHE["field_names_ordered"] = [id2name[fid] for fid in get_field_order_by_domain()]
    return _TAXONOMY_CACHE["field_names_ordered"]


def get_subfields_for_field(field_id: int) -> List[int]:
    """Return ordered list of subfield IDs belonging to a field."""
    df = _ensure_taxonomy_loaded()
    return sorted(df.loc[df["field_id"] == field_id, "subfield_id"].drop_duplicates().tolist())


def get_all_field_subfield_map() -> Dict[int, List[int]]:
    """Return {field_id: [subfield_ids]} for all fields."""
    if "field_subfield_map" not in _TAXONOMY_CACHE:
        df = _ensure_taxonomy_loaded()[["field_id", "subfield_id"]].drop_duplicates()
        result = {}
        for fid in df["field_id"].unique():
            result[int(fid)] = sorted(df.loc[df["field_id"] == fid, "subfield_id"].tolist())
        _TAXONOMY_CACHE["field_subfield_map"] = result
    return _TAXONOMY_CACHE["field_subfield_map"]


# ============================================================================
# COLOR FUNCTIONS
# ============================================================================

def get_domain_color(domain: int | str) -> str:
    """Get color for a domain by ID or name."""
    return DOMAIN_COLORS.get(domain, DOMAIN_COLORS["Other"])


def get_field_color(field: int | str) -> str:
    """Get color for a field (based on its parent domain)."""
    if isinstance(field, str):
        field = get_field_name_to_id().get(field, -1)
    dom_id = get_field_id_to_domain_id().get(field, -1)
    return get_domain_color(dom_id)


def get_subfield_color(subfield_id: int) -> str:
    """Get color for a subfield (based on its grandparent domain)."""
    dom_id = get_subfield_id_to_domain_id().get(int(subfield_id), -1)
    return get_domain_color(dom_id)


def darken_hex(hex_color: str, factor: float = 0.65) -> str:
    """Darken a hex color by a factor (0-1). Used for ISITE overlay bars."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join([c * 2 for c in h])
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return "#5a5a5a"
    r, g, b = int(r * factor), int(g * factor), int(b * factor)
    r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple (for WordCloud)."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join([c * 2 for c in h])
    try:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return (127, 127, 127)


# ============================================================================
# BLOB PARSERS — BASIC
# ============================================================================

def parse_pipe_int_list(blob: str) -> List[int]:
    """Parse '101 | 86 | 118' -> [101, 86, 118]."""
    if pd.isna(blob) or not str(blob).strip():
        return []
    return [safe_int(x) for x in str(blob).split("|")]


def parse_pipe_float_list(blob: str) -> List[float]:
    """Parse '1.5 | 0.8 | 2.1' -> [1.5, 0.8, 2.1]."""
    if pd.isna(blob) or not str(blob).strip():
        return []
    return [safe_float(x) for x in str(blob).split("|")]


def parse_pipe_str_list(blob: str) -> List[str]:
    """Parse 'Name1 | Name2 | Name3' -> ['Name1', 'Name2', 'Name3']."""
    if pd.isna(blob) or not str(blob).strip():
        return []
    return [x.strip() for x in str(blob).split("|")]


def parse_pipe_bool_list(blob: str) -> List[bool]:
    """Parse 'True | False | True' -> [True, False, True]."""
    if pd.isna(blob) or not str(blob).strip():
        return []
    return [x.strip().lower() in ("true", "1", "yes") for x in str(blob).split("|")]


def parse_parallel_lists(cols_config: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    """
    Parse multiple parallel pipe-separated lists into a DataFrame.
    
    Args:
        cols_config: {output_col_name: (blob_value, type)} where type is 'str', 'int', 'float', 'bool'
    """
    parsed = {}
    max_len = 0
    
    for col_name, (blob, dtype) in cols_config.items():
        if dtype == "int":
            values = parse_pipe_int_list(blob)
        elif dtype == "float":
            values = parse_pipe_float_list(blob)
        elif dtype == "bool":
            values = parse_pipe_bool_list(blob)
        else:
            values = parse_pipe_str_list(blob)
        parsed[col_name] = values
        max_len = max(max_len, len(values))
    
    for col_name, (blob, dtype) in cols_config.items():
        fill = 0 if dtype == "int" else (np.nan if dtype == "float" else (False if dtype == "bool" else ""))
        while len(parsed[col_name]) < max_len:
            parsed[col_name].append(fill)
    
    return pd.DataFrame(parsed)


# ============================================================================
# BLOB PARSERS — STRUCTURED
# ============================================================================

def parse_year_domain_blob(blob: str) -> pd.DataFrame:
    """
    Parse 'Pubs per year per domain' blob.
    Format: '2019 (14 ; 19 ; 0 ; 68) | 2020 (12 ; 25 ; 1 ; 55) | ...'
    Domain order in parentheses: 1, 2, 3, 4 (Life, Social, Physical, Health)
    
    Returns DataFrame[year, domain_id, domain_name, count, color].
    """
    if pd.isna(blob) or not str(blob).strip():
        return pd.DataFrame(columns=["year", "domain_id", "domain_name", "count", "color"])
    
    rows = []
    dom_id2name = get_domain_id_to_name()
    
    for part in str(blob).split("|"):
        part = part.strip()
        m = re.match(r"^\s*(\d{4})\s*\((.*?)\)\s*$", part)
        if not m:
            continue
        year = int(m.group(1))
        values = [safe_int(x) for x in m.group(2).split(";")]
        
        for i, dom_id in enumerate(DOMAIN_ORDER):
            count = values[i] if i < len(values) else 0
            rows.append({
                "year": year,
                "domain_id": dom_id,
                "domain_name": dom_id2name.get(dom_id, f"Domain {dom_id}"),
                "count": count,
                "color": get_domain_color(dom_id),
            })
    
    return pd.DataFrame(rows)


def parse_positional_field_counts(blob: str) -> pd.DataFrame:
    """
    Parse 'Pubs per field' blob (positional, IDs 11-36).
    Format: '3 | 3 | 15 | 2 | 0 | ...' (26 values)
    
    Returns DataFrame[field_id, field_name, count, domain_id, domain_name, color].
    Ordered by domain grouping.
    """
    values = parse_pipe_int_list(blob)
    id2name = get_field_id_to_name()
    id2dom = get_field_id_to_domain_id()
    dom2name = get_domain_id_to_name()
    field_order = get_field_order_by_domain()
    
    # Build lookup: field_id -> count (data is stored in ID order 11-36)
    field_counts = {}
    for i, count in enumerate(values):
        field_id = 11 + i
        if field_id in id2name:
            field_counts[field_id] = count
    
    # Build rows in domain-grouped order
    rows = []
    for field_id in field_order:
        if field_id not in id2name:
            continue
        dom_id = id2dom.get(field_id, 0)
        rows.append({
            "field_id": field_id,
            "field_name": id2name[field_id],
            "count": field_counts.get(field_id, 0),
            "domain_id": dom_id,
            "domain_name": dom2name.get(dom_id, "Other"),
            "color": get_field_color(field_id),
        })
    
    return pd.DataFrame(rows)


def parse_positional_domain_counts(blob: str) -> pd.DataFrame:
    """
    Parse 'Pubs per domain' blob (positional, IDs 1-4).
    Format: '70 | 198 | 8 | 286' (4 values)
    
    Returns DataFrame[domain_id, domain_name, count, color].
    """
    values = parse_pipe_int_list(blob)
    dom2name = get_domain_id_to_name()
    
    rows = []
    for i, dom_id in enumerate(DOMAIN_ORDER):
        count = values[i] if i < len(values) else 0
        rows.append({
            "domain_id": dom_id,
            "domain_name": dom2name.get(dom_id, f"Domain {dom_id}"),
            "count": count,
            "color": get_domain_color(dom_id),
        })
    
    return pd.DataFrame(rows)


def parse_fwci_boxplot_blob(blob: str) -> pd.DataFrame:
    """
    Parse 'FWCI boxplot per field id (centiles 0,10,25,50,75,90,100)' blob.
    Format: '11 (0.00 ; 0.10 ; 0.32 ; 0.92 ; 1.44 ; 5.40 ; 12.5) | 12 (...) | ...'
    
    Returns DataFrame[field_id, field_name, p0, p10, p25, p50, p75, p90, p100, domain_id, domain_name, color].
    Ordered by domain grouping.
    """
    id2name = get_field_id_to_name()
    id2dom = get_field_id_to_domain_id()
    dom2name = get_domain_id_to_name()
    field_order = get_field_order_by_domain()
    
    # Parse blob into dict
    field_data = {}
    if not pd.isna(blob) and str(blob).strip():
        for part in str(blob).split("|"):
            part = part.strip()
            m = re.match(r"^\s*(\d+)\s*\((.*?)\)\s*$", part)
            if not m:
                continue
            field_id = int(m.group(1))
            values = [safe_float(x) for x in m.group(2).split(";")]
            if len(values) < 7:
                values.extend([np.nan] * (7 - len(values)))
            field_data[field_id] = values
    
    # Build rows for ALL fields in domain-grouped order
    rows = []
    for field_id in field_order:
        if field_id not in id2name:
            continue
        dom_id = id2dom.get(field_id, 0)
        
        if field_id in field_data:
            values = field_data[field_id]
        else:
            values = [np.nan] * 7
        
        rows.append({
            "field_id": field_id,
            "field_name": id2name.get(field_id, f"Field {field_id}"),
            "p0": values[0],
            "p10": values[1],
            "p25": values[2],
            "p50": values[3],
            "p75": values[4],
            "p90": values[5],
            "p100": values[6],
            "domain_id": dom_id,
            "domain_name": dom2name.get(dom_id, "Other"),
            "color": get_field_color(field_id),
        })
    
    return pd.DataFrame(rows)


def parse_subfield_column(blob: str, field_id: int) -> pd.DataFrame:
    """
    Parse a single 'Pubs per subfield within "X" (id: Y)' column.
    Format: '0 | 5 | 12 | ...' (positional by subfield ID within that field)
    
    Returns DataFrame[subfield_id, subfield_name, count, color].
    """
    values = parse_pipe_int_list(blob)
    subfields = get_subfields_for_field(field_id)
    sub2name = get_subfield_id_to_name()
    
    rows = []
    for i, count in enumerate(values):
        if i >= len(subfields):
            break
        sub_id = subfields[i]
        rows.append({
            "subfield_id": sub_id,
            "subfield_name": sub2name.get(sub_id, f"Subfield {sub_id}"),
            "count": count,
            "color": get_subfield_color(sub_id),
        })
    
    return pd.DataFrame(rows)


# ============================================================================
# UTILITIES
# ============================================================================

def pad_dataframe(df: pd.DataFrame, n_rows: int, numeric_cols: List[str] | None = None) -> pd.DataFrame:
    """
    Ensure DataFrame has exactly n_rows. Truncate if more, pad with blanks if fewer.
    Numeric columns get NaN, text columns get empty string.
    """
    if len(df) >= n_rows:
        return df.head(n_rows).reset_index(drop=True)
    
    numeric_cols = set(numeric_cols or [])
    missing = n_rows - len(df)
    
    filler = {col: (np.nan if col in numeric_cols else "") for col in df.columns}
    filler_df = pd.DataFrame([filler] * missing)
    return pd.concat([df, filler_df], ignore_index=True)


def build_openalex_url(openalex_id: str) -> str:
    """Build OpenAlex URL from ID."""
    if pd.isna(openalex_id) or not str(openalex_id).strip():
        return ""
    oid = str(openalex_id).strip()
    if not oid.startswith("http"):
        return f"https://openalex.org/{oid}"
    return oid

def render_domain_legend():
    """Render inline domain color legend."""
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:16px;">'
        f'<span style="width:14px;height:14px;background:{get_domain_color(d)};border-radius:3px;margin-right:6px;"></span>'
        f'{d}</span>'
        for d in DOMAIN_NAMES_ORDERED
    )
    st.markdown(f'<div style="margin:8px 0 16px 0;">{items}</div>', unsafe_allow_html=True)