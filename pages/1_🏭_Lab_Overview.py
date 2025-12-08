# pages/1_🏭_Lab_View.py
"""
Lab View: Single lab analysis with precomputed indicators from ul_labs.parquet.
"""
from __future__ import annotations

import re
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Local imports
from lib.data_cache import get_labs_df, get_topics_df
from lib.helpers import (
    # Constants
    YEARS, DOMAIN_ORDER, DOMAIN_NAMES_ORDERED,
    # Taxonomy
    init_taxonomy, get_domain_id_to_name, get_field_id_to_name,
    get_field_order_by_domain, get_subfields_for_field,
    get_subfield_id_to_name, get_subfield_id_to_field_id, get_subfield_id_to_domain_id,
    get_field_id_to_domain_id,
    # Colors
    get_domain_color, get_field_color, get_subfield_color, darken_hex, hex_to_rgb,
    # Parsers
    safe_int, safe_float,
    parse_pipe_int_list, parse_pipe_float_list, parse_pipe_str_list, parse_pipe_bool_list,
    parse_parallel_lists, parse_year_domain_blob, parse_positional_field_counts,
    parse_fwci_boxplot_blob, parse_subfield_column,
    # Utilities
    pad_dataframe, build_openalex_url,
    # Legend
    render_domain_legend
)

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(page_title="Lab View", page_icon="🏭", layout="wide")

# Initialize taxonomy cache
init_taxonomy(get_topics_df())

# ============================================================================
# CONSTANTS
# ============================================================================

# Emoji markers for domains (for tables)
DOMAIN_EMOJI = {
    "Health Sciences": "🟥",
    "Life Sciences": "🟩",
    "Physical Sciences": "🟦",
    "Social Sciences": "🟨",
    "Other": "⬜",
}

# Document type colors
DOCTYPE_COLORS = {
    "Articles": "#4285F4",
    "Book chapters": "#FBBC05",
    "Books": "#EA4335",
    "Reviews": "#34A853",
    "Preprints": "#9E9E9E",
}

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def get_all_structures_df() -> pd.DataFrame:
    """Load all structures (not just labs) from ul_labs.parquet."""
    from lib.data_cache import get_labs_df
    # Reload without the lab filter - need to modify or use raw loading
    from pathlib import Path
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    return pd.read_parquet(DATA_DIR / "ul_labs.parquet")


df_all_structures = get_all_structures_df()
structure_types = df_all_structures["Structure type"].dropna().unique().tolist()

if df_all_structures.empty:
    st.error("No structures found in the data.")
    st.stop()


# ============================================================================
# HELPER: Build OpenAlex Works URL
# ============================================================================

def build_openalex_works_url(openalex_id: str) -> str:
    """
    Build OpenAlex Works search URL from institution ID.
    """
    if pd.isna(openalex_id) or not str(openalex_id).strip():
        return ""
    oid = str(openalex_id).strip()
    if "/" in oid:
        oid = oid.split("/")[-1]
    return (
        f"https://openalex.org/works?page=1&filter="
        f"authorships.institutions.lineage:{oid},"
        f"publication_year:2019-2023,"
        f"type:types/article|types/book-chapter|types/book|types/review"
    )


# ============================================================================
# LAB-SPECIFIC TABLE BUILDERS
# ============================================================================

def build_field_distribution_table(row: pd.Series, pubs_total: int) -> pd.DataFrame:
    """
    Build table for field distribution bar chart with ISITE overlay.
    Returns DataFrame ordered by domain.
    """
    df_total = parse_positional_field_counts(row.get("Pubs per field", ""))
    df_isite = parse_positional_field_counts(row.get("ISITE pubs per field", ""))
    
    if df_total.empty:
        return pd.DataFrame()
    
    df = df_total.copy()
    if not df_isite.empty:
        df = df.merge(
            df_isite[["field_id", "count"]].rename(columns={"count": "isite_count"}),
            on="field_id", how="left"
        )
    else:
        df["isite_count"] = 0
    
    df["isite_count"] = df["isite_count"].fillna(0).astype(int)
    pubs_total = max(1, pubs_total)
    df["share"] = df["count"] / pubs_total
    df["isite_share"] = df["isite_count"] / pubs_total
    df["color_dark"] = df["color"].apply(lambda c: darken_hex(c, 0.65))
    
    return df


def build_fwci_whisker_table(row: pd.Series) -> pd.DataFrame:
    """
    Build table for FWCI whisker plot with counts.
    Returns DataFrame ordered by domain, including ALL fields.
    """
    df = parse_fwci_boxplot_blob(row.get("FWCI boxplot per field id (centiles 0,10,25,50,75,90,100)", ""))
    
    if df.empty:
        # Create empty frame with all fields
        field_order = get_field_order_by_domain()
        id2name = get_field_id_to_name()
        id2dom = get_field_id_to_domain_id()
        dom2name = get_domain_id_to_name()
        rows = []
        for field_id in field_order:
            if field_id not in id2name:
                continue
            dom_id = id2dom.get(field_id, 0)
            rows.append({
                "field_id": field_id,
                "field_name": id2name[field_id],
                "p0": np.nan, "p10": np.nan, "p25": np.nan, "p50": np.nan,
                "p75": np.nan, "p90": np.nan, "p100": np.nan,
                "count": 0,
                "domain_id": dom_id,
                "domain_name": dom2name.get(dom_id, "Other"),
                "color": get_field_color(field_id),
            })
        df = pd.DataFrame(rows)
    
    # Add counts from Pubs per field
    df_counts = parse_positional_field_counts(row.get("Pubs per field", ""))
    if not df_counts.empty:
        count_map = dict(zip(df_counts["field_id"], df_counts["count"]))
        df["count"] = df["field_id"].map(count_map).fillna(0).astype(int)
    else:
        if "count" not in df.columns:
            df["count"] = 0
    
    return df


def build_subfield_wordcloud_data(row: pd.Series) -> pd.DataFrame:
    """
    Aggregate all subfield counts for wordcloud.
    Returns DataFrame[subfield_id, subfield_name, count, color] with count > 0.
    """
    all_rows = []
    sub2dom = get_subfield_id_to_domain_id()
    sub2name = get_subfield_id_to_name()
    
    for field_id in range(11, 37):
        col_pattern = f'Pubs per subfield within .* \\(id: {field_id}\\)'
        matching_cols = [c for c in row.index if re.match(col_pattern, c)]
        if not matching_cols:
            continue
        
        blob = row.get(matching_cols[0], "")
        values = parse_pipe_int_list(blob)
        subfields = get_subfields_for_field(field_id)
        
        for i, count in enumerate(values):
            if i >= len(subfields) or count <= 0:
                continue
            sub_id = subfields[i]
            dom_id = sub2dom.get(sub_id, 0)
            all_rows.append({
                "subfield_id": sub_id,
                "subfield_name": sub2name.get(sub_id, f"Subfield {sub_id}"),
                "count": count,
                "color": get_domain_color(dom_id),
            })
    
    if not all_rows:
        return pd.DataFrame(columns=["subfield_id", "subfield_name", "count", "color"])
    
    df = pd.DataFrame(all_rows)
    df = df.groupby(["subfield_id", "subfield_name", "color"], as_index=False)["count"].sum()
    return df.sort_values("count", ascending=False).reset_index(drop=True)


def build_subfield_table(row: pd.Series, pubs_total: int) -> pd.DataFrame:
    """
    Build detailed subfield table with counts, ratios vs UL, and FWCI.
    Returns DataFrame with all subfields that have count > 0.
    """
    sub2name = get_subfield_id_to_name()
    sub2field = get_subfield_id_to_field_id()
    sub2dom = get_subfield_id_to_domain_id()
    field2name = get_field_id_to_name()
    dom2name = get_domain_id_to_name()
    
    all_rows = []
    
    for field_id in range(11, 37):
        # Find columns for this field
        count_pattern = f'Pubs per subfield within .* \\(id: {field_id}\\)'
        ratio_pattern = f'Ratio against UL.*\\(id: {field_id}\\)'
        fwci_pattern = f'FWCI per subfield within .* \\(id: {field_id}\\)'
        
        count_cols = [c for c in row.index if re.match(count_pattern, c)]
        ratio_cols = [c for c in row.index if re.match(ratio_pattern, c)]
        fwci_cols = [c for c in row.index if re.match(fwci_pattern, c)]
        
        if not count_cols:
            continue
        
        counts = parse_pipe_int_list(row.get(count_cols[0], ""))
        ratios = parse_pipe_float_list(row.get(ratio_cols[0], "")) if ratio_cols else []
        fwcis = parse_pipe_float_list(row.get(fwci_cols[0], "")) if fwci_cols else []
        
        subfields = get_subfields_for_field(field_id)
        
        for i, sub_id in enumerate(subfields):
            count = counts[i] if i < len(counts) else 0
            if count <= 0:
                continue
            
            dom_id = sub2dom.get(sub_id, 0)
            dom_name = dom2name.get(dom_id, "Other")
            
            # Get ratio - ensure we get the right value
            ratio_val = ratios[i] if i < len(ratios) else np.nan
            fwci_val = fwcis[i] if i < len(fwcis) else np.nan
            
            all_rows.append({
                "subfield_id": sub_id,
                "Subfield": sub2name.get(sub_id, f"Subfield {sub_id}"),
                "Field": field2name.get(field_id, f"Field {field_id}"),
                "Domain": dom_name,
                "Domain marker": f"{DOMAIN_EMOJI.get(dom_name, '⬜')} {dom_name}",
                "count": count,
                "share_of_lab": count / max(1, pubs_total),
                "ratio_vs_ul": ratio_val,
                "fwci": fwci_val,
            })
    
    if not all_rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_rows)
    return df.sort_values("count", ascending=False).reset_index(drop=True)


def parse_internal_collabs_blob(blob: str) -> pd.DataFrame:
    """
    Parse 'Top 10 internal lab/other collabs (type,count,ratio,FWCI)' blob.
    Format: 'Name (type, count ; ratio ; fwci) | ...'
    """
    if pd.isna(blob) or not str(blob).strip() or str(blob).strip().lower() == "none":
        return pd.DataFrame(columns=["name", "type", "count", "ratio", "fwci"])
    
    rows = []
    pattern = re.compile(r"^\s*(.+?)\s*\((lab|other),\s*(\d+)\s*;\s*([\d.]+)\s*;\s*([\d.]+)\)\s*$", re.IGNORECASE)
    
    for part in str(blob).split("|"):
        m = pattern.match(part.strip())
        if m:
            rows.append({
                "name": m.group(1).strip(),
                "type": m.group(2).lower(),
                "count": safe_int(m.group(3)),
                "ratio": safe_float(m.group(4)),
                "fwci": safe_float(m.group(5)),
            })
    
    return pd.DataFrame(rows)


def build_authors_table(row: pd.Series) -> pd.DataFrame:
    """Build table for top 10 authors."""
    df = parse_parallel_lists({
        "Author": (row.get("Top 10 authors (name)", ""), "str"),
        "Pubs": (row.get("Top 10 authors (pubs)", ""), "int"),
        "Avg FWCI (FR)": (row.get("Top 10 authors (Average FWCI_FR)", ""), "float"),
        "_is_lor": (row.get("Top 10 authors (Is Lorraine)", ""), "bool"),
        "Other UL affiliations": (row.get("Top 10 authors (Other internal affiliation(s))", ""), "str"),
    })
    df["Is UL"] = df["_is_lor"].apply(lambda x: "Yes" if x else "No")
    df["Other UL affiliations"] = df["Other UL affiliations"].str.replace(";", ", ")
    return df[["Author", "Pubs", "Avg FWCI (FR)", "Is UL", "Other UL affiliations"]]


def build_international_partners_table(row: pd.Series) -> pd.DataFrame:
    """Build table for top 10 international partners with FWCI."""
    df = parse_parallel_lists({
        "Partner": (row.get("Top 10 int partners (name)", ""), "str"),
        "Type": (row.get("Top 10 int partners (type)", ""), "str"),
        "Country": (row.get("Top 10 int partners (country)", ""), "str"),
        "Co-pubs": (row.get("Top 10 int partners (copubs with structure)", ""), "int"),
        "% of UL copubs": (row.get("Top 10 int partners (% of all UL copubs with this partner)", ""), "float"),
        "Avg FWCI": (row.get("Top 10 int partners (FWCI)", ""), "float"),
    })
    df["% of UL copubs"] = df["% of UL copubs"] * 100
    return df


def build_french_partners_table(row: pd.Series) -> pd.DataFrame:
    """Build table for top 10 French partners with FWCI."""
    df = parse_parallel_lists({
        "Partner": (row.get("Top 10 FR partners (name)", ""), "str"),
        "Type": (row.get("Top 10 FR partners (type)", ""), "str"),
        "Co-pubs": (row.get("Top 10 FR partners (copubs with lab)", ""), "int"),
        "% of UL copubs": (row.get("Top 10 FR partners (% of all UL copubs with this partner)", ""), "float"),
        "Avg FWCI": (row.get("Top 10 FR partners (FWCI)", ""), "float"),
    })
    df["% of UL copubs"] = df["% of UL copubs"] * 100
    return df


def parse_doctype_blob(blob: str) -> Dict[str, int]:
    """
    Parse 'Pubs per type (articles | book chapters | books | reviews | preprints)' blob.
    Format: '460 | 57 | 3 | 33 | 9'
    """
    values = parse_pipe_int_list(blob)
    labels = ["Articles", "Book chapters", "Books", "Reviews", "Preprints"]
    result = {}
    for i, label in enumerate(labels):
        result[label] = values[i] if i < len(values) else 0
    return result


# ============================================================================
# PLOTLY CHART BUILDERS
# ============================================================================

def plot_doctype_pie(row: pd.Series) -> go.Figure:
    """
    Create pie chart for document type distribution.
    Uses 'Pubs per type (articles | book chapters | books | reviews | preprints)' column.
    """
    blob = row.get("Pubs per type (articles | book chapters | books | reviews | preprints)", "")
    doc_counts = parse_doctype_blob(blob)
    
    labels = list(doc_counts.keys())
    values = list(doc_counts.values())
    colors = [DOCTYPE_COLORS.get(l, "#9E9E9E") for l in labels]
    
    # Filter out zero values
    data = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=250, margin=dict(t=10, b=10, l=10, r=10))
        return fig
    
    labels_f, values_f, colors_f = zip(*data)
    
    fig = go.Figure(data=[go.Pie(
        labels=labels_f,
        values=values_f,
        marker=dict(colors=colors_f, line=dict(color='white', width=2)),
        textinfo='percent',
        textposition='inside',
        insidetextorientation='horizontal',
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
        hole=0.0,
    )])
    
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=10, b=60, l=10, r=10),
        height=280,
    )
    
    return fig


def plot_yearly_stacked_by_domain(df_yr_dom: pd.DataFrame, title: str = "") -> go.Figure:
    """Stacked bar chart of publications by year and domain."""
    if df_yr_dom.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Create color map from domain names
    color_map = {d: get_domain_color(d) for d in DOMAIN_NAMES_ORDERED}
    
    fig = px.bar(
        df_yr_dom,
        x="year",
        y="count",
        color="domain_name",
        color_discrete_map=color_map,
        category_orders={"domain_name": DOMAIN_NAMES_ORDERED},
        title=title,
        labels={"count": "Publications", "year": "Year", "domain_name": "Domain"},
    )
    fig.update_layout(
        barmode="stack",
        xaxis=dict(tickmode="array", tickvals=YEARS, ticktext=[str(y) for y in YEARS]),
        yaxis=dict(title="Publications (count)"),
        showlegend=False,
        template="plotly_white",
        margin=dict(t=40, b=40),
    )
    return fig


def plot_field_distribution(df_fields: pd.DataFrame, title: str = "") -> go.Figure:
    """Horizontal bar chart of field distribution with ISITE overlay and count gutter."""
    if df_fields.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Calculate gutter size
    max_share = float(df_fields["share"].max() or 0.0)
    if max_share <= 0:
        max_share = 0.01
    gutter = max_share * 0.20
    
    fig = go.Figure()
    
    # Total bars
    fig.add_trace(go.Bar(
        y=df_fields["field_name"],
        x=df_fields["share"],
        orientation="h",
        marker_color=df_fields["color"].tolist(),
        name="Total",
        hovertemplate="<b>%{y}</b><br>Total: %{x:.1%}<extra></extra>",
    ))
    
    # ISITE overlay bars
    fig.add_trace(go.Bar(
        y=df_fields["field_name"],
        x=df_fields["isite_share"],
        orientation="h",
        marker_color=df_fields["color_dark"].tolist(),
        name="ISITE",
        width=0.5,
        hovertemplate="<b>%{y}</b><br>ISITE: %{x:.1%}<extra></extra>",
    ))
    
    # Add count annotations in the gutter
    for field_name, cnt in zip(df_fields["field_name"], df_fields["count"]):
        fig.add_annotation(
            x=-gutter * 0.95,
            y=field_name,
            text=f"{int(cnt):,}",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=11, color="#444"),
        )
    
    fig.update_layout(
        barmode="overlay",
        title=title,
        xaxis=dict(
            title="% of lab publications",
            tickformat=".0%",
            range=[-gutter, max_share * 1.10],
            showgrid=True,
            gridcolor="#e0e0e0",
        ),
        yaxis=dict(autorange="reversed", title="", tickfont=dict(size=12)),
        showlegend=False,
        template="plotly_white",
        margin=dict(l=10, r=10, t=60, b=40),
        height=max(450, len(df_fields) * 22 + 120),
    )
    return fig


def plot_fwci_whiskers(df_fwci: pd.DataFrame, title: str = "") -> go.Figure:
    """
    Horizontal box-whisker plot for FWCI distribution per field.
    Uses centiles 0, 25, 50, 75, 100 only.
    Shows ALL fields including those with 0 publications.
    """
    if df_fwci.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Calculate max for this lab only - find max p100 among fields with data
    valid_p100 = df_fwci.loc[df_fwci["count"] > 0, "p100"].dropna()
    if valid_p100.empty:
        xmax = 5.0
    else:
        xmax = float(valid_p100.max())
        if xmax <= 0 or np.isnan(xmax):
            xmax = 5.0
    
    # Gutter for count labels
    gutter = xmax * 0.15
    
    fig = go.Figure()
    
    for i, row in df_fwci.iterrows():
        y = row["field_name"]
        color = row["color"]
        count = row["count"]
        
        # Draw elements only if count > 0 and we have valid median
        if count > 0 and pd.notna(row["p50"]):
            # Whisker line (p0 to p100)
            if pd.notna(row["p0"]) and pd.notna(row["p100"]):
                fig.add_trace(go.Scatter(
                    x=[row["p0"], row["p100"]],
                    y=[y, y],
                    mode="lines",
                    line=dict(color=color, width=2),
                    showlegend=False,
                    hoverinfo="skip",
                ))
            
            # Box (p25 to p75) using a bar
            if pd.notna(row["p25"]) and pd.notna(row["p75"]) and row["p75"] >= row["p25"]:
                fig.add_trace(go.Bar(
                    x=[row["p75"] - row["p25"]],
                    y=[y],
                    base=row["p25"],
                    orientation="h",
                    marker=dict(color=color, opacity=0.3),
                    width=0.6,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{y}</b><br>"
                        f"Min: {row['p0']:.2f}<br>"
                        f"Q1: {row['p25']:.2f}<br>"
                        f"Median: {row['p50']:.2f}<br>"
                        f"Q3: {row['p75']:.2f}<br>"
                        f"Max: {row['p100']:.2f}<extra></extra>"
                    ),
                ))
            
            # Median marker
            if pd.notna(row["p50"]):
                fig.add_trace(go.Scatter(
                    x=[row["p50"]],
                    y=[y],
                    mode="markers",
                    marker=dict(color=color, size=10, symbol="line-ns", line=dict(width=3, color=color)),
                    showlegend=False,
                    hoverinfo="skip",
                ))
    
    # Add count annotations in gutter for ALL fields
    for i, row in df_fwci.iterrows():
        fig.add_annotation(
            x=-gutter * 0.95,
            y=row["field_name"],
            text=f"{int(row['count']):,}",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=11, color="#444"),
        )
    
    fig.update_layout(
        title=title,
        xaxis=dict(
            title="FWCI (France)",
            range=[-gutter, xmax * 1.10],
            showgrid=True,
            gridcolor="#e0e0e0",
        ),
        yaxis=dict(autorange="reversed", title="", tickfont=dict(size=12)),
        showlegend=False,
        template="plotly_white",
        margin=dict(l=10, r=10, t=60, b=40),
        height=max(450, len(df_fwci) * 22 + 120),
        barmode="overlay",
    )
    return fig


def render_subfield_wordcloud(df_sub: pd.DataFrame, title: str = ""):
    """Render subfield wordcloud using WordCloud library."""
    if df_sub.empty:
        st.info("No subfield data available.")
        return
    
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
    except ImportError:
        st.info("Install `wordcloud` package to see the subfield wordcloud.")
        return
    
    freqs = dict(zip(df_sub["subfield_name"], df_sub["count"]))
    name2color = dict(zip(df_sub["subfield_name"], df_sub["color"]))
    
    def color_func(word, *args, **kwargs):
        return name2color.get(word, "#7f7f7f")
    
    wc = WordCloud(width=900, height=350, background_color="white", prefer_horizontal=0.9)
    wc.generate_from_frequencies(freqs)
    wc.recolor(color_func=color_func)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, pad=10)
    plt.tight_layout()
    st.pyplot(fig)


# ============================================================================
# PAGE LAYOUT
# ============================================================================

st.title("🏭 Lab View")

# -----------------------------------------------------------------------------
# Section 1: Overview Table with Type Filter
# -----------------------------------------------------------------------------

st.subheader("Overview per Internal Structure (2019–2023)")

# Type filter
selected_types = st.multiselect(
    "Filter by structure type",
    options=structure_types,
    default=structure_types,
    key="structure_type_filter"
)

# Filter data
df_filtered = df_all_structures[df_all_structures["Structure type"].isin(selected_types)].copy()

if df_filtered.empty:
    st.warning("No structures match the selected filters.")
else:
    summary = df_filtered[[
        "Structure name", "Structure type", "Pole", "Pubs total",
        "Pubs PPtop10% (subfield)", "Pubs PPtop1% (subfield)",
        "Pubs ISITE (In_LUE)", "Pubs international", "Pubs with company",
    ]].copy()
    
    summary = summary.rename(columns={
        "Structure name": "Structure",
        "Structure type": "Type",
        "Pubs total": "Publications",
        "Pubs PPtop10% (subfield)": "Top 10%",
        "Pubs PPtop1% (subfield)": "Top 1%",
        "Pubs ISITE (In_LUE)": "Pubs ISITE",
    })
    
    # Compute percentages (max value = 1 for all progress bars)
    summary["% ISITE"] = summary["Pubs ISITE"] / summary["Publications"].replace(0, np.nan)
    summary["% international"] = summary["Pubs international"] / summary["Publications"].replace(0, np.nan)
    summary["% with company"] = summary["Pubs with company"] / summary["Publications"].replace(0, np.nan)
    
    summary = summary.sort_values("Publications", ascending=False)
    
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
        column_order=[
            "Structure", "Type", "Pole", "Publications", "Top 10%", "Top 1%",
            "Pubs ISITE", "% ISITE", "% international", "% with company",
        ],
        column_config={
            "Structure": st.column_config.TextColumn("Structure"),
            "Type": st.column_config.TextColumn("Type"),
            "Pole": st.column_config.TextColumn("Pole"),
            "Publications": st.column_config.NumberColumn("Publications", format="%.0f"),
            "Top 10%": st.column_config.NumberColumn("Top 10%", format="%.0f"),
            "Top 1%": st.column_config.NumberColumn("Top 1%", format="%.0f"),
            "Pubs ISITE": st.column_config.NumberColumn("Pubs ISITE", format="%.0f"),
            "% ISITE": st.column_config.ProgressColumn("% ISITE", format="%.1f%%", min_value=0, max_value=1),
            "% international": st.column_config.ProgressColumn("% international", format="%.1f%%", min_value=0, max_value=1),
            "% with company": st.column_config.ProgressColumn("% with company", format="%.1f%%", min_value=0, max_value=1),
        },
    )

st.divider()

# -----------------------------------------------------------------------------
# Section 2: Single Lab Analysis
# -----------------------------------------------------------------------------

st.subheader("Structure Analysis")

structure_names = df_all_structures["Structure name"].dropna().astype(str).sort_values().tolist()
selected_structure = st.selectbox("Select a structure", structure_names, index=0)
row = df_all_structures.loc[df_all_structures["Structure name"] == selected_structure].iloc[0]
pubs_total = safe_int(row.get("Pubs total", 0))

# --- Structure Profile Section ---
st.markdown("---")

col_info, col_pie, col_metrics = st.columns([2, 1, 1])

with col_info:
    st.markdown(f"### {selected_structure}")
    st.write(f"**Type:** {row.get('Structure type', '—')}")
    st.write(f"**Pole:** {row.get('Pole', '—')}")
    st.write(f"**ROR:** {row.get('ROR', '—')}")
    
    # OpenAlex link
    oa_url = build_openalex_works_url(row.get("OpenAlex ID", ""))
    if oa_url:
        st.markdown(f"[🔗 View in OpenAlex]({oa_url})")
    
    # Big number display
    st.markdown(
        f"<div style='font-size:48px;font-weight:bold;color:#E63946;margin:20px 0;'>{pubs_total:,}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        """<div style='color:#666;font-size:14px;'>
        <strong>Total publications (2019–2023)</strong><br/>
        Includes articles, reviews, book chapters, books, and some preprints 
        (with a DOI, an abstract available, and no similar existing published manuscript).
        </div>""",
        unsafe_allow_html=True
    )

with col_pie:
    st.markdown("**Document types**")
    fig_pie = plot_doctype_pie(row)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_metrics:
    st.markdown("**Key indicators**")
    
    pubs_isite = safe_int(row.get("Pubs ISITE (In_LUE)", 0))
    pct_isite = (pubs_isite / pubs_total * 100) if pubs_total > 0 else 0
    st.metric("ISITE publications", f"{pubs_isite:,}", f"{pct_isite:.1f}%")
    
    pubs_top10 = safe_int(row.get("Pubs PPtop10% (subfield)", 0))
    pct_top10 = (pubs_top10 / pubs_total * 100) if pubs_total > 0 else 0
    st.metric("Top 10%", f"{pubs_top10:,}", f"{pct_top10:.1f}%")
    
    pubs_top1 = safe_int(row.get("Pubs PPtop1% (subfield)", 0))
    pct_top1 = (pubs_top1 / pubs_total * 100) if pubs_total > 0 else 0
    st.metric("Top 1%", f"{pubs_top1:,}", f"{pct_top1:.1f}%")
    
    pubs_intl = safe_int(row.get("Pubs international", 0))
    pct_intl = (pubs_intl / pubs_total * 100) if pubs_total > 0 else 0
    st.metric("International", f"{pubs_intl:,}", f"{pct_intl:.1f}%")
    
    pubs_company = safe_int(row.get("Pubs with company", 0))
    pct_company = (pubs_company / pubs_total * 100) if pubs_total > 0 else 0
    st.metric("With industry", f"{pubs_company:,}", f"{pct_company:.1f}%")

st.markdown("---")

# --- Domain Legend ---
render_domain_legend()

# --- Yearly Distribution by Domain ---
st.markdown("#### Yearly Distribution by Domain")
df_year_dom = parse_year_domain_blob(row.get("Pubs per year per domain", ""))
fig_yearly = plot_yearly_stacked_by_domain(df_year_dom)
st.plotly_chart(fig_yearly, use_container_width=True)

# --- Subfield Word Cloud ---
st.markdown("#### Subfields Word Cloud")
df_sub = build_subfield_wordcloud_data(row)
render_subfield_wordcloud(df_sub)

# --- Field Distribution ---
st.markdown("#### Field Distribution (% of structure) — Total & ISITE")
st.markdown("*Darker bars represent publications from the ISITE initiative.*")
df_fields = build_field_distribution_table(row, pubs_total)
fig_fields = plot_field_distribution(df_fields)
st.plotly_chart(fig_fields, use_container_width=True)

# --- FWCI Whiskers ---
st.markdown("#### FWCI Distribution by Field (vs France)")
df_fwci = build_fwci_whisker_table(row)
fig_fwci = plot_fwci_whiskers(df_fwci)
st.plotly_chart(fig_fwci, use_container_width=True)

st.markdown("---")

# --- Subfield Detail Table ---
st.markdown("#### Subfield Detail")

df_subfield_table = build_subfield_table(row, pubs_total)

if df_subfield_table.empty:
    st.info("No subfield-level data for this structure.")
else:
    df_sub_display = df_subfield_table[[
        "Domain marker", "Field", "Subfield", "count", "share_of_lab", "ratio_vs_ul", "fwci"
    ]].rename(columns={
        "Domain marker": "Domain",
        "count": "Publications",
        "share_of_lab": "% of structure total",
        "ratio_vs_ul": "% of UL in this subfield",
        "fwci": "Avg FWCI",
    })
    
    st.dataframe(
        df_sub_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Domain": st.column_config.TextColumn("Domain"),
            "Field": st.column_config.TextColumn("Field"),
            "Subfield": st.column_config.TextColumn("Subfield"),
            "Publications": st.column_config.NumberColumn("Publications", format="%d"),
            "% of structure total": st.column_config.ProgressColumn(
                "% of structure total",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
                help="Fraction of this structure's publications in this subfield.",
            ),
            "% of UL in this subfield": st.column_config.ProgressColumn(
                "% of UL in this subfield",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
                help="Share of UL's total publications in this subfield that come from this structure.",
            ),
            "Avg FWCI": st.column_config.NumberColumn("Avg FWCI", format="%.2f"),
        },
    )

st.markdown("---")

# --- Top 10 Internal Partners ---
st.markdown("#### Top 10 Internal Partners")
df_collabs = parse_internal_collabs_blob(row.get("Top 10 internal lab/other collabs (type,count,ratio,FWCI)", ""))
df_collabs = pad_dataframe(df_collabs.head(10), 10, numeric_cols=["count", "ratio", "fwci"])
df_collabs["% of structure pubs"] = df_collabs["ratio"]

st.dataframe(
    df_collabs,
    use_container_width=True,
    hide_index=True,
    column_order=["name", "type", "count", "% of structure pubs", "fwci"],
    column_config={
        "name": st.column_config.TextColumn("Partner"),
        "type": st.column_config.TextColumn("Type"),
        "count": st.column_config.NumberColumn("Co-pubs", format="%.0f"),
        "% of structure pubs": st.column_config.ProgressColumn(
            "% of structure pubs",
            format="%.1f%%",
            min_value=0,
            max_value=1
        ),
        "fwci": st.column_config.NumberColumn("Avg FWCI", format="%.2f"),
    },
)

st.markdown("---")

# --- Top 10 International Partners ---
st.markdown("#### Top 10 International Partners")
pct_intl_display = safe_int(row.get("Pubs international", 0)) / max(1, pubs_total) * 100
st.metric("% international publications", f"{pct_intl_display:.1f}%")

df_intl = build_international_partners_table(row)
df_intl = pad_dataframe(df_intl, 10, numeric_cols=["Co-pubs", "% of UL copubs", "Avg FWCI"])

st.dataframe(
    df_intl,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Partner": st.column_config.TextColumn("Partner"),
        "Country": st.column_config.TextColumn("Country"),
        "Co-pubs": st.column_config.NumberColumn("Co-pubs", format="%.0f"),
        "% of UL copubs": st.column_config.ProgressColumn("% of UL copubs", format="%.1f%%", min_value=0, max_value=100),
        "Type": st.column_config.TextColumn("Type"),
        "Avg FWCI": st.column_config.NumberColumn("Avg FWCI", format="%.2f"),
    },
)

st.markdown("---")

# --- Top 10 French Partners ---
st.markdown("#### Top 10 French Partners")
df_fr = build_french_partners_table(row)
df_fr = pad_dataframe(df_fr, 10, numeric_cols=["Co-pubs", "% of UL copubs", "Avg FWCI"])

st.dataframe(
    df_fr,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Partner": st.column_config.TextColumn("Partner"),
        "Co-pubs": st.column_config.NumberColumn("Co-pubs", format="%.0f"),
        "% of UL copubs": st.column_config.ProgressColumn("% of UL copubs", format="%.1f%%", min_value=0, max_value=100),
        "Type": st.column_config.TextColumn("Type"),
        "Avg FWCI": st.column_config.NumberColumn("Avg FWCI", format="%.2f"),
    },
)

st.markdown("---")

# --- Top 10 Authors ---
st.markdown("#### Top 10 Authors")
df_authors = build_authors_table(row)
df_authors = pad_dataframe(df_authors, 10, numeric_cols=["Pubs", "Avg FWCI (FR)"])

st.dataframe(
    df_authors,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Author": st.column_config.TextColumn("Author"),
        "Pubs": st.column_config.NumberColumn("Pubs", format="%.0f"),
        "Avg FWCI (FR)": st.column_config.NumberColumn("Avg FWCI (FR)", format="%.3f"),
        "Is UL": st.column_config.TextColumn("Is UL"),
        "Other UL affiliations": st.column_config.TextColumn("Other UL affiliations"),
    },
)