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


@st.cache_data
def load_thematic_overview():
    return pd.read_parquet("data/thematic_overview.parquet")

@st.cache_data
def load_thematic_sublevels():
    return pd.read_parquet("data/thematic_detail_sublevels.parquet")

@st.cache_data
def load_thematic_contributions():
    return pd.read_parquet("data/thematic_detail_contributions.parquet")

@st.cache_data
def load_thematic_partners():
    return pd.read_parquet("data/thematic_detail_partners.parquet")

@st.cache_data
def load_thematic_authors():
    return pd.read_parquet("data/thematic_detail_authors.parquet")

@st.cache_data
def load_lab_info():
    """Load lab names and types from structures file."""
    try:
        df = pd.read_parquet("data/ul_labs.parquet")
        # Index by structure_key (which is the ROR in the contributions file)
        return df.set_index("structure_key")[["Structure name", "Structure type"]].to_dict("index")
    except Exception as e:
        st.warning(f"Could not load lab info: {e}")
        return {}

@st.cache_data
def load_partners_base():
    """Load partner base data for reciprocity calculations."""
    try:
        return pd.read_parquet("data/ul_partners_base.parquet")
    except Exception as e:
        st.warning(f"Could not load partners base: {e}")
        return pd.DataFrame()

@st.cache_data
def load_tm_labels():
    """Load topic model labels and keywords."""
    try:
        return pd.read_parquet("data/TM_labels.parquet")
    except Exception as e:
        return pd.DataFrame()