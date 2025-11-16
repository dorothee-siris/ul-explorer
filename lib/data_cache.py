# lib/data_cache.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

@st.cache_resource
def get_core_df() -> pd.DataFrame:
    """All UL publications (20 MB parquet)."""
    return pd.read_parquet(DATA_DIR / "pubs.parquet")

@st.cache_resource
def get_partners_df() -> pd.DataFrame:
    """Full partners table (5 MB parquet) with light type tweaks."""
    df = pd.read_parquet(DATA_DIR / "ul_partners.parquet")

    # Light memory-friendly tweaks that are useful everywhere
    for col in ["Partner name", "Country", "Partner type"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df

@st.cache_resource
def get_topics_df() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "all_topics.parquet")

@st.cache_resource
def get_lookup_df() -> pd.DataFrame:
    # if you need it
    return pd.read_parquet(DATA_DIR / "ul_lookup.parquet")

# Add this function to lib/data_cache.py
@st.cache_resource
def get_labs_df() -> pd.DataFrame:
    """Laboratory structures data (300 KB parquet)."""
    df = pd.read_parquet(DATA_DIR / "ul_labs.parquet")
    # Only return labs, not other structure types
    return df[df["Structure type"] == "lab"].copy()