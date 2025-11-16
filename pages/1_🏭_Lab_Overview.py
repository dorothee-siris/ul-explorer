from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Import taxonomy helpers
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "lib"))

from lib.taxonomy import (
    build_taxonomy_lookups,
    get_domain_color,
    get_field_color,
    get_domain_for_field,
    canonical_field_order,
)

# ---------------------------- Configuration ----------------------------

UNITS_PATH = REPO_ROOT / "data" / "ul_labs.parquet"
YEAR_START, YEAR_END = 2019, 2023

# ------------------------------- Data Loading -------------------------------

from lib.data_cache import get_labs_df

# --------------------------- Parsing Helpers ---------------------------

def parse_year_domain_blob(blob: str) -> pd.DataFrame:
    """
    Parse year/domain data from format: '2019 (14 ; 19 ; 0 ; 68) | 2020 (19 ; 21 ; 1 ; ...)'
    Returns DataFrame with columns: year, domain, count
    """
    if pd.isna(blob) or not str(blob).strip():
        return pd.DataFrame(columns=["year", "domain", "count"])
    
    rows = []
    # Domain mapping by position index in the blob
    domain_order = ["Health Sciences", "Life Sciences", "Physical Sciences", "Social Sciences"]
    
    for year_part in str(blob).split("|"):
        year_part = year_part.strip()
        m = re.match(r"(\d{4})\s*\((.*?)\)", year_part)
        if not m:
            continue
        
        year = int(m.group(1))
        counts = m.group(2).split(";")
        
        for idx, count_str in enumerate(counts):
            if idx < len(domain_order):
                count = int(count_str.strip()) if count_str.strip().isdigit() else 0
                if count > 0:
                    rows.append((year, domain_order[idx], count))
    
    return pd.DataFrame(rows, columns=["year", "domain", "count"])

def parse_field_blob(blob: str, taxonomy: Dict) -> pd.DataFrame:
    """
    Parse field data from '15 | 26 | 17 | ...' format
    Returns DataFrame with field_name and count
    """
    if pd.isna(blob) or not str(blob).strip():
        return pd.DataFrame(columns=["field_name", "count"])
    
    field_counts = {}
    for field_id in str(blob).split("|"):
        field_id = field_id.strip()
        if field_id:
            field_name = taxonomy["id2name"].get(field_id, f"Field {field_id}")
            field_counts[field_name] = field_counts.get(field_name, 0) + 1
    
    return pd.DataFrame(
        list(field_counts.items()), 
        columns=["field_name", "count"]
    ).sort_values("count", ascending=False)

def parse_subfield_columns(row: pd.Series, taxonomy: Dict) -> pd.DataFrame:
    """
    Parse all subfield columns and aggregate counts
    Returns DataFrame with: name, count, field, domain, color
    """
    subfield_data = []
    
    for col in row.index:
        if "Copubs per subfield within" not in col:
            continue
            
        # Extract parent field ID from column name
        m = re.search(r'\(id:\s*(\d+)\)', col)
        if not m:
            continue
        
        field_id = m.group(1)
        field_name = taxonomy["id2name"].get(field_id, f"Field {field_id}")
        domain = get_domain_for_field(field_id)
        
        # Parse subfield IDs and counts
        blob = row[col]
        if pd.isna(blob) or not str(blob).strip():
            continue
        
        for item in str(blob).split("|"):
            item = item.strip()
            if not item:
                continue
            
            # Extract subfield ID (first number in the item)
            subfield_id = item.split()[0] if item.split() else item
            subfield_name = taxonomy["id2name"].get(subfield_id, f"Subfield {subfield_id}")
            
            subfield_data.append({
                "name": subfield_name,
                "count": 1,
                "field": field_name,
                "domain": domain,
                "color": get_domain_color(domain)
            })
    
    if not subfield_data:
        return pd.DataFrame(columns=["name", "count", "field", "domain", "color"])
    
    # Aggregate by subfield name
    df = pd.DataFrame(subfield_data)
    return df.groupby(["name", "field", "domain", "color"], as_index=False)["count"].sum()

def parse_fwci_blob(blob: str, taxonomy: Dict) -> Dict[str, float]:
    """
    Parse FWCI data from 'field_id | field_id | ...' format
    Returns dict of {field_name: fwci_value}
    """
    if pd.isna(blob) or not str(blob).strip():
        return {}
    
    fwci_values = {}
    items = str(blob).split("|")
    
    for item in items:
        item = item.strip()
        if not item:
            continue
        
        # Try to parse as "field_id" or extract from more complex format
        parts = item.split()
        if parts and parts[0].isdigit():
            field_id = parts[0]
            field_name = taxonomy["id2name"].get(field_id, f"Field {field_id}")
            # Extract FWCI value if present (assume last numeric value)
            fwci = 1.0  # Default
            for p in reversed(parts):
                try:
                    fwci = float(p)
                    break
                except ValueError:
                    continue
            fwci_values[field_name] = fwci
    
    return fwci_values

# ------------------------------ Visualization Functions ------------------------------

def create_yearly_domain_chart(df: pd.DataFrame) -> go.Figure:
    """Create interactive stacked bar chart for yearly publications by domain"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No yearly data available", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Get domain colors from taxonomy
    domain_colors = {d: get_domain_color(d) for d in df["domain"].unique()}
    
    fig = px.bar(
        df,
        x="year",
        y="count",
        color="domain",
        color_discrete_map=domain_colors,
        barmode="stack",
        labels={
            "year": "Year",
            "count": "Publications",
            "domain": "Domain",
        },
        hover_data={"count": ":,"}
    )
    
    fig.update_layout(
        margin=dict(l=0, r=10, t=40, b=10),
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        hovermode="x unified"
    )
    
    return fig

def create_field_distribution_chart(df: pd.DataFrame, taxonomy: Dict) -> go.Figure:
    """Create horizontal bar chart for field distribution"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No field data available", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Order fields according to canonical order
    canon_order = canonical_field_order()
    ordered_fields = [f for f in canon_order if f in df["field_name"].values]
    
    # Add any fields not in canonical order
    extra_fields = [f for f in df["field_name"].values if f not in canon_order]
    ordered_fields.extend(extra_fields)
    
    df_ordered = pd.DataFrame({"field_name": ordered_fields})
    df_ordered = df_ordered.merge(df, on="field_name", how="left").fillna(0)
    
    # Add colors based on domain
    df_ordered["color"] = df_ordered["field_name"].apply(get_field_color)
    df_ordered["domain"] = df_ordered["field_name"].apply(get_domain_for_field)
    
    fig = go.Figure(go.Bar(
        y=df_ordered["field_name"],
        x=df_ordered["count"],
        orientation="h",
        marker=dict(color=df_ordered["color"]),
        text=df_ordered["count"].astype(int),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Publications: %{x}<br>Domain: %{customdata}<extra></extra>",
        customdata=df_ordered["domain"]
    ))
    
    fig.update_layout(
        xaxis_title="Number of Publications",
        yaxis_title="",
        height=max(400, len(df_ordered) * 25),
        margin=dict(l=200, r=50, t=20, b=50),
        showlegend=False,
        hovermode="closest"
    )
    
    fig.update_yaxis(autorange="reversed")
    
    return fig

def create_subfield_treemap(df: pd.DataFrame) -> go.Figure:
    """Create interactive treemap for subfields"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No subfield data available", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Create hierarchical structure
    labels = ["All"] + df["domain"].unique().tolist() + df["name"].tolist()
    parents = [""] + ["All"] * len(df["domain"].unique()) + df["domain"].tolist()
    values = [df["count"].sum()] + [
        df[df["domain"] == d]["count"].sum() for d in df["domain"].unique()
    ] + df["count"].tolist()
    
    # Colors
    colors = ["#ffffff"] + [
        get_domain_color(d) for d in df["domain"].unique()
    ] + df["color"].tolist()
    
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=colors),
        textinfo="label+value",
        hovertemplate="<b>%{label}</b><br>Publications: %{value}<br>Parent: %{parent}<extra></extra>"
    ))
    
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=20, b=0)
    )
    
    return fig

# ------------------------------ Main UI ------------------------------

st.set_page_config(page_title="Lab View · Analysis", layout="wide")
st.title("🏭 Laboratory Analysis Dashboard")

# Load data
taxonomy = build_taxonomy_lookups()
df_labs = get_labs_df()

if df_labs.empty:
    st.error("No laboratory data found.")
    st.stop()

# Laboratory selection
st.header("Select Laboratory")

lab_names = df_labs["Structure name"].dropna().astype(str).sort_values().tolist()
default_lab = "LRGP" if "LRGP" in lab_names else lab_names[0]
default_idx = lab_names.index(default_lab) if default_lab in lab_names else 0

selected_lab = st.selectbox(
    "Choose a laboratory for analysis:",
    lab_names,
    index=default_idx,
    key="lab_select"
)

# Get selected lab data
lab_data = df_labs[df_labs["Structure name"] == selected_lab].iloc[0]

st.divider()

# ----------------------------- Overview Section -----------------------------

st.header(f"📊 {selected_lab} - Overview")

# Key metrics
col1, col2, col3, col4, col5 = st.columns(5)

total_pubs = int(lab_data.get("Copubs total", 0))
lue_pubs = int(lab_data.get("Pubs ISITE (In_LUE)", 0))
top10 = int(lab_data.get("Pubs PPtop10% (subfield)", 0))
top1 = int(lab_data.get("Pubs PPtop1% (subfield)", 0))
intl = int(lab_data.get("Pubs international", 0))
company = int(lab_data.get("Pubs with company", 0))

with col1:
    st.metric("Total Publications", f"{total_pubs:,}")
with col2:
    st.metric("LUE Publications", f"{lue_pubs:,}")
with col3:
    st.metric("Top 10% Papers", f"{top10:,}")
with col4:
    st.metric("Top 1% Papers", f"{top1:,}")
with col5:
    st.metric("International", f"{intl:,}")

# Additional metrics
col6, col7, col8, col9 = st.columns(4)

with col6:
    st.metric("Industry Collab", f"{company:,}")
with col7:
    pct_intl = (intl / total_pubs * 100) if total_pubs > 0 else 0
    st.metric("% International", f"{pct_intl:.1f}%")
with col8:
    pct_company = (company / total_pubs * 100) if total_pubs > 0 else 0
    st.metric("% Industry", f"{pct_company:.1f}%")
with col9:
    pct_lue = (lue_pubs / total_pubs * 100) if total_pubs > 0 else 0
    st.metric("% LUE", f"{pct_lue:.1f}%")

st.divider()

# ----------------------------- Visualizations -----------------------------

st.header("📈 Research Output Analysis")

# Yearly distribution by domain
st.subheader("Publications by Year and Domain")
df_year_dom = parse_year_domain_blob(lab_data.get("Copubs per year per domain", ""))
if not df_year_dom.empty:
    fig_year = create_yearly_domain_chart(df_year_dom)
    st.plotly_chart(fig_year, use_container_width=True)
else:
    st.info("No yearly/domain breakdown available.")

# Field distribution
st.subheader("Publications by Research Field")
df_fields = parse_field_blob(lab_data.get("Copubs per field", ""), taxonomy)
if not df_fields.empty:
    fig_fields = create_field_distribution_chart(df_fields, taxonomy)
    st.plotly_chart(fig_fields, use_container_width=True)
else:
    st.info("No field distribution data available.")

# Subfield treemap
st.subheader("Research Subfields Distribution")
df_subfields = parse_subfield_columns(lab_data, taxonomy)
if not df_subfields.empty:
    fig_sub = create_subfield_treemap(df_subfields)
    st.plotly_chart(fig_sub, use_container_width=True)
else:
    st.info("No subfield data available.")

st.divider()

# ----------------------------- Partnerships -----------------------------

st.header("🤝 Collaborations & Partnerships")

# Helper function to parse partner data
def parse_partners(names_str: str, types_str: str, countries_str: str, copubs_str: str) -> pd.DataFrame:
    """Parse partner data from pipe-separated strings"""
    if pd.isna(names_str) or not str(names_str).strip():
        return pd.DataFrame()
    
    names = [n.strip() for n in str(names_str).split("|")]
    types = [t.strip() for t in str(types_str).split("|")] if pd.notna(types_str) else []
    countries = [c.strip() for c in str(countries_str).split("|")] if pd.notna(countries_str) else []
    copubs = [int(c.strip()) if c.strip().isdigit() else 0 
              for c in str(copubs_str).split("|")] if pd.notna(copubs_str) else []
    
    # Pad lists to same length
    max_len = len(names)
    types += [""] * (max_len - len(types))
    countries += [""] * (max_len - len(countries))
    copubs += [0] * (max_len - len(copubs))
    
    return pd.DataFrame({
        "Partner": names[:10],  # Top 10 only
        "Type": types[:10],
        "Country": countries[:10] if countries else None,
        "Co-publications": copubs[:10]
    }).dropna(axis=1, how='all')

# International partners
st.subheader("Top 10 International Partners")
df_int = parse_partners(
    lab_data.get("Top 10 int partners (name)", ""),
    lab_data.get("Top 10 int partners (type)", ""),
    lab_data.get("Top 10 int partners (country)", ""),
    lab_data.get("Top 10 int partners (copubs with structure)", "")
)

if not df_int.empty:
    st.dataframe(df_int, use_container_width=True, hide_index=True)
else:
    st.info("No international partnership data available.")

# French partners
st.subheader("Top 10 French Partners")
df_fr = parse_partners(
    lab_data.get("Top 10 FR partners (name)", ""),
    lab_data.get("Top 10 FR partners (type)", ""),
    "",  # No country column for French partners
    lab_data.get("Top 10 FR partners (copubs with lab)", "")
)

if not df_fr.empty:
    st.dataframe(df_fr, use_container_width=True, hide_index=True)
else:
    st.info("No French partnership data available.")

st.divider()

# ----------------------------- Authors -----------------------------

st.header("👥 Top Authors")

# Parse authors data
def parse_authors(names_str: str, pubs_str: str, orcids_str: str, 
                  fwci_str: str, lorraine_str: str) -> pd.DataFrame:
    """Parse author data from pipe-separated strings"""
    if pd.isna(names_str) or not str(names_str).strip():
        return pd.DataFrame()
    
    names = [n.strip() for n in str(names_str).split("|")]
    pubs = [int(p.strip()) if p.strip().isdigit() else 0 
            for p in str(pubs_str).split("|")] if pd.notna(pubs_str) else []
    orcids = [o.strip() for o in str(orcids_str).split("|")] if pd.notna(orcids_str) else []
    fwci = [float(f.strip()) if f.strip().replace(".", "").replace("-", "").isdigit() else 0.0 
            for f in str(fwci_str).split("|")] if pd.notna(fwci_str) else []
    lorraine = ["Yes" if l.strip().lower() == "true" else "No" 
                for l in str(lorraine_str).split("|")] if pd.notna(lorraine_str) else []
    
    # Pad lists
    max_len = len(names)
    pubs += [0] * (max_len - len(pubs))
    orcids += [""] * (max_len - len(orcids))
    fwci += [0.0] * (max_len - len(fwci))
    lorraine += [""] * (max_len - len(lorraine))
    
    return pd.DataFrame({
        "Author": names[:10],
        "Publications": pubs[:10],
        "ORCID": orcids[:10],
        "Avg FWCI": fwci[:10],
        "UL Affiliated": lorraine[:10]
    })

df_authors = parse_authors(
    lab_data.get("Top 10 authors (name)", ""),
    lab_data.get("Top 10 authors (pubs)", ""),
    lab_data.get("Top 10 authors (Orcid)", ""),
    lab_data.get("Top 10 authors (Average FWCI_FR)", ""),
    lab_data.get("Top 10 authors (Is Lorraine)", "")
)

if not df_authors.empty:
    st.dataframe(
        df_authors,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Publications": st.column_config.NumberColumn(format="%d"),
            "Avg FWCI": st.column_config.NumberColumn(format="%.2f"),
        }
    )
else:
    st.info("No author data available.")

# ----------------------------- Footer Info -----------------------------

st.divider()
col_id1, col_id2 = st.columns(2)
with col_id1:
    if "OpenAlex ID" in lab_data.index and pd.notna(lab_data["OpenAlex ID"]):
        st.markdown(f"**OpenAlex ID:** `{lab_data['OpenAlex ID']}`")
with col_id2:
    if "ROR" in lab_data.index and pd.notna(lab_data["ROR"]):
        st.markdown(f"**ROR ID:** `{lab_data['ROR']}`")