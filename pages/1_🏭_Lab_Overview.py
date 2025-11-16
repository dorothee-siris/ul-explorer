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

@st.cache_data(show_spinner=False)
def load_labs_data() -> pd.DataFrame:
    """Load and filter laboratory data"""
    df = pd.read_parquet(UNITS_PATH)
    return df[df["Structure type"] == "lab"].copy()

@st.cache_data(show_spinner=False)
def get_taxonomy() -> Dict:
    """Get taxonomy lookups"""
    return build_taxonomy_lookups()

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
            "count": "Number of publications",
            "domain": "",
        },
    )
    
    fig.update_layout(
        margin=dict(l=0, r=10, t=10, b=10),
        showlegend=False,
        height=380,
        hovermode="x unified"
    )
    
    fig.update_xaxes(tickfont=dict(size=12))
    fig.update_yaxes(tickfont=dict(size=12))
    
    return fig

def create_field_distribution_chart(df: pd.DataFrame, taxonomy: Dict) -> go.Figure:
    """Create horizontal bar chart for field distribution with count annotations"""
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
    
    # Calculate percentage share
    total = df_ordered["count"].sum()
    df_ordered["share_pct"] = (df_ordered["count"] / total * 100) if total > 0 else 0
    
    fig = px.bar(
        df_ordered,
        x="share_pct",
        y="field_name",
        orientation="h",
        color="domain",
        color_discrete_map={d: get_domain_color(d) for d in df_ordered["domain"].unique()},
        labels={"share_pct": "Share (%)", "field_name": ""},
        custom_data=["count", "domain"],
    )
    
    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Share: %{x:.1f}%<br>"
            "Publications: %{customdata[0]:,.0f}<br>"
            "Domain: %{customdata[1]}"
            "<extra></extra>"
        ),
    )
    
    max_share = float(df_ordered["share_pct"].max() or 0.0)
    if max_share <= 0:
        max_share = 1.0
    gutter = max_share * 0.20
    
    fig.update_xaxes(
        range=[-gutter, max_share * 1.05],
        showgrid=True,
        gridcolor="#e0e0e0",
        ticksuffix="%",
        tickfont=dict(size=12),
    )
    fig.update_yaxes(tickfont=dict(size=13))
    
    # Add count annotations in gutter
    for field_name, cnt in zip(df_ordered["field_name"], df_ordered["count"]):
        fig.add_annotation(
            x=-gutter * 0.98,
            y=field_name,
            text=f"{int(cnt)}",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=12, color="#444"),
        )
    
    fig.update_layout(
        margin=dict(l=0, r=10, t=25, b=10),
        showlegend=False,
        height=600,
    )
    
    return fig

def create_subfield_treemap(df: pd.DataFrame) -> go.Figure:
    """Create interactive treemap for subfields"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No subfield data available", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Aggregate by domain first for better hierarchy
    domain_totals = df.groupby("domain")["count"].sum().to_dict()
    
    # Create hierarchical data with proper structure
    fig = go.Figure(go.Treemap(
        labels=df["name"].tolist() + list(domain_totals.keys()) + ["All"],
        parents=df["domain"].tolist() + ["All"] * len(domain_totals) + [""],
        values=df["count"].tolist() + list(domain_totals.values()) + [df["count"].sum()],
        marker=dict(
            colors=df["color"].tolist() + [get_domain_color(d) for d in domain_totals.keys()] + ["#ffffff"],
            line=dict(width=2)
        ),
        textinfo="label+value",
        hovertemplate="<b>%{label}</b><br>Publications: %{value}<br>Parent: %{parent}<extra></extra>",
        textfont=dict(size=12)
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
taxonomy = get_taxonomy()
df_labs = load_labs_data()

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

# Department/Pole info if available
if "Pole" in lab_data.index and pd.notna(lab_data["Pole"]):
    st.markdown(f"**Pôle:** {lab_data['Pole']}")

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

# Create HTML legend for domains (reusable)
def create_domain_legend() -> str:
    """Create HTML legend for domain colors"""
    domains = ["Health Sciences", "Life Sciences", "Physical Sciences", "Social Sciences"]
    legend_html = "<div style='margin: 0.8rem 0 0.4rem 0;'>"
    for d in domains:
        color = get_domain_color(d)
        legend_html += (
            f"<span style='display:inline-block;width:12px;height:12px;"
            f"border-radius:50%;background-color:{color};margin-right:4px;'></span>"
            f"<span style='margin-right:14px;'>{d}</span>"
        )
    legend_html += "</div>"
    return legend_html

# Yearly distribution by domain
st.subheader("Publications by Year and Domain")
st.markdown(create_domain_legend(), unsafe_allow_html=True)
df_year_dom = parse_year_domain_blob(lab_data.get("Copubs per year per domain", ""))
if not df_year_dom.empty:
    fig_year = create_yearly_domain_chart(df_year_dom)
    st.plotly_chart(fig_year, use_container_width=True)
else:
    st.info("No yearly/domain breakdown available.")

# Field distribution
st.subheader("Publications by Research Field")
st.markdown(create_domain_legend(), unsafe_allow_html=True)
df_fields = parse_field_blob(lab_data.get("Copubs per field", ""), taxonomy)
if not df_fields.empty:
    fig_fields = create_field_distribution_chart(df_fields, taxonomy)
    st.plotly_chart(fig_fields, use_container_width=True)
else:
    st.info("No field distribution data available.")

# Subfield treemap
st.subheader("Research Subfields Distribution")
st.markdown(create_domain_legend(), unsafe_allow_html=True)
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
    st.dataframe(
        df_int, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Partner": st.column_config.TextColumn("Partner", width="large"),
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Country": st.column_config.TextColumn("Country", width="medium"),
            "Co-publications": st.column_config.NumberColumn(
                "Co-publications", 
                format="%d",
                help="Number of co-publications with this partner"
            ),
        }
    )
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
    st.dataframe(
        df_fr, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Partner": st.column_config.TextColumn("Partner", width="large"),
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Co-publications": st.column_config.NumberColumn(
                "Co-publications", 
                format="%d",
                help="Number of co-publications with this partner"
            ),
        }
    )
else:
    st.info("No French partnership data available.")

st.divider()

# ----------------------------- Authors -----------------------------

st.header("👥 Top Authors")

# Parse authors data
def parse_authors(names_str: str, pubs_str: str, orcids_str: str, 
                  fwci_str: str, lorraine_str: str, other_affil_str: str = "") -> pd.DataFrame:
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
    other_affil = [a.strip() for a in str(other_affil_str).split("|")] if pd.notna(other_affil_str) else []
    
    # Pad lists
    max_len = len(names)
    pubs += [0] * (max_len - len(pubs))
    orcids += [""] * (max_len - len(orcids))
    fwci += [0.0] * (max_len - len(fwci))
    lorraine += [""] * (max_len - len(lorraine))
    other_affil += [""] * (max_len - len(other_affil))
    
    df = pd.DataFrame({
        "Author": names[:10],
        "Publications": pubs[:10],
        "ORCID": orcids[:10],
        "Avg FWCI": fwci[:10],
        "UL Affiliated": lorraine[:10],
        "Other affiliations": other_affil[:10]
    })
    
    # Clean up Other affiliations (remove empty or just semicolons)
    df["Other affiliations"] = df["Other affiliations"].apply(
        lambda x: x.replace(";", ", ").strip(", ") if x else ""
    )
    
    return df

df_authors = parse_authors(
    lab_data.get("Top 10 authors (name)", ""),
    lab_data.get("Top 10 authors (pubs)", ""),
    lab_data.get("Top 10 authors (Orcid)", ""),
    lab_data.get("Top 10 authors (Average FWCI_FR)", ""),
    lab_data.get("Top 10 authors (Is Lorraine)", ""),
    lab_data.get("Top 10 authors (Other internal affiliation(s))", "")
)

if not df_authors.empty:
    st.dataframe(
        df_authors,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Author": st.column_config.TextColumn("Author", width="medium"),
            "Publications": st.column_config.NumberColumn(
                "Publications", 
                format="%d",
                help="Number of publications in this lab"
            ),
            "Avg FWCI": st.column_config.NumberColumn(
                "Avg FWCI", 
                format="%.2f",
                help="Average Field-Weighted Citation Impact"
            ),
            "ORCID": st.column_config.TextColumn("ORCID", width="small"),
            "UL Affiliated": st.column_config.TextColumn("UL Affiliated", width="small"),
            "Other affiliations": st.column_config.TextColumn(
                "Other UL affiliations", 
                width="medium",
                help="Other labs or structures within UL"
            ),
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