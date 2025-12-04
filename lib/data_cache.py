# lib/data_cache.py
"""
Centralized data loading with Streamlit caching.
All parquet files are loaded once and shared across views.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


@st.cache_resource
def get_topics_df() -> pd.DataFrame:
    """Taxonomy: domains, fields, subfields, topics (all_topics.parquet)."""
    return pd.read_parquet(DATA_DIR / "all_topics.parquet")


@st.cache_resource
def get_labs_df() -> pd.DataFrame:
    """Laboratory structures with precomputed indicators (ul_labs.parquet)."""
    df = pd.read_parquet(DATA_DIR / "ul_labs.parquet")
    return df[df["Structure type"] == "lab"].copy()


@st.cache_resource
def get_partners_df() -> pd.DataFrame:
    """Full partners table with type optimizations."""
    df = pd.read_parquet(DATA_DIR / "ul_partners.parquet")
    for col in ["Partner name", "Country", "Partner type"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


@st.cache_resource
def get_core_df() -> pd.DataFrame:
    """All UL publications (pubs.parquet) - only load if needed."""
    return pd.read_parquet(DATA_DIR / "pubs.parquet")


@st.cache_resource
def get_lookup_df() -> pd.DataFrame:
    """Lookup table if needed."""
    return pd.read_parquet(DATA_DIR / "ul_lookup.parquet")