# pages/1_🏭_Lab_View.py
"""
Lab View: Single lab analysis with precomputed indicators from ul_labs.parquet.
"""
from __future__ import annotations

import re
from typing import List

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
    # Colors
    get_domain_color, get_field_color, get_subfield_color, darken_hex, hex_to_rgb,
    # Parsers
    safe_int, safe_float,
    parse_pipe_int_list, parse_pipe_float_list, parse_pipe_str_list, parse_pipe_bool_list,
    parse_parallel_lists, parse_year_domain_blob, parse_positional_field_counts,
    parse_fwci_boxplot_blob, parse_subfield_column,
    # Utilities
    pad_dataframe, build_openalex_url,
)

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(page_title="Lab View", page_icon="🏭", layout="wide")

# Initialize taxonomy cache
init_taxonomy(get_topics_df())

# ============================================================================
# DATA LOADING
# ============================================================================

df_labs = get_labs_df()
lab_names = df_labs["Structure name"].dropna().astype(str).sort_values().tolist()

if not lab_names:
    st.error("No labs found in the data.")
    st.stop()


# ============================================================================
# LAB-SPECIFIC TABLE BUILDERS
# ============================================================================

def build_field_distribution_table(row: pd.Series, pubs_total: int) -> pd.DataFrame:
    """
    Build table for field distribution bar chart with ISITE overlay.
    Returns DataFrame ordered by domain with columns:
        field_id, field_name, count, isite_count, share, isite_share, color, color_dark, domain_id, domain_name
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
    Returns DataFrame ordered by domain.
    """
    df = parse_fwci_boxplot_blob(row.get("FWCI boxplot per field id (centiles 0,10,25,50,75,90,100)", ""))
    
    if df.empty:
        return df
    
    # Add counts from Pubs per field
    df_counts = parse_positional_field_counts(row.get("Pubs per field", ""))
    if not df_counts.empty:
        df = df.merge(df_counts[["field_id", "count"]], on="field_id", how="left")
    else:
        df["count"] = 0
    
    df["count"] = df["count"].fillna(0).astype(int)
    return df


def build_subfield_wordcloud_data(row: pd.Series) -> pd.DataFrame:
    """
    Aggregate all subfield counts for wordcloud.
    Returns DataFrame[subfield_id, subfield_name, count, color] with count > 0.
    """
    all_rows = []
    
    for field_id in range(11, 37):
        col_pattern = f'Pubs per subfield within .* \\(id: {field_id}\\)'
        matching_cols = [c for c in row.index if re.match(col_pattern, c)]
        if not matching_cols:
            continue
        df_sub = parse_subfield_column(row.get(matching_cols[0], ""), field_id)
        if not df_sub.empty:
            all_rows.append(df_sub)
    
    if not all_rows:
        return pd.DataFrame(columns=["subfield_id", "subfield_name", "count", "color"])
    
    df = pd.concat(all_rows, ignore_index=True)
    df = df[df["count"] > 0].copy()
    df = df.groupby(["subfield_id", "subfield_name", "color"], as_index=False)["count"].sum()
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
    """Build table for top 10 international partners."""
    df = parse_parallel_lists({
        "Partner": (row.get("Top 10 int partners (name)", ""), "str"),
        "Type": (row.get("Top 10 int partners (type)", ""), "str"),
        "Country": (row.get("Top 10 int partners (country)", ""), "str"),
        "Co-pubs": (row.get("Top 10 int partners (copubs with structure)", ""), "int"),
        "% of UL copubs": (row.get("Top 10 int partners (% of all UL copubs with this partner)", ""), "float"),
    })
    df["% of UL copubs"] = df["% of UL copubs"] * 100
    return df


def build_french_partners_table(row: pd.Series) -> pd.DataFrame:
    """Build table for top 10 French partners."""
    df = parse_parallel_lists({
        "Partner": (row.get("Top 10 FR partners (name)", ""), "str"),
        "Type": (row.get("Top 10 FR partners (type)", ""), "str"),
        "Co-pubs": (row.get("Top 10 FR partners (copubs with lab)", ""), "int"),
        "% of UL copubs": (row.get("Top 10 FR partners (% of all UL copubs with this partner)", ""), "float"),
    })
    df["% of UL copubs"] = df["% of UL copubs"] * 100
    return df


# ============================================================================
# PLOTLY CHART BUILDERS
# ============================================================================

def plot_yearly_stacked_by_domain(df_yr_dom: pd.DataFrame, title: str = "") -> go.Figure:
    """Stacked bar chart of publications by year and domain."""
    if df_yr_dom.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = px.bar(
        df_yr_dom,
        x="year",
        y="count",
        color="domain_name",
        color_discrete_map={d: get_domain_color(d) for d in DOMAIN_NAMES_ORDERED},
        category_orders={"domain_name": DOMAIN_NAMES_ORDERED},
        title=title,
        labels={"count": "Publications", "year": "Year", "domain_name": "Domain"},
    )
    fig.update_layout(
        barmode="stack",
        xaxis=dict(tickmode="array", tickvals=YEARS, ticktext=[str(y) for y in YEARS]),
        yaxis=dict(title="Publications (count)"),
        legend_title_text="Domain",
        template="plotly_white",
        margin=dict(t=40, b=40),
    )
    return fig


def plot_field_distribution(df_fields: pd.DataFrame, title: str = "") -> go.Figure:
    """Horizontal bar chart of field distribution with ISITE overlay."""
    if df_fields.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure()
    
    # Total bars
    fig.add_trace(go.Bar(
        y=df_fields["field_name"],
        x=df_fields["share"],
        orientation="h",
        marker_color=df_fields["color"],
        name="Total",
        text=[f"{c:,}" for c in df_fields["count"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Total: %{x:.1%}<br>Count: %{text}<extra></extra>",
    ))
    
    # ISITE overlay bars
    fig.add_trace(go.Bar(
        y=df_fields["field_name"],
        x=df_fields["isite_share"],
        orientation="h",
        marker_color=df_fields["color_dark"],
        name="ISITE",
        width=0.5,
        hovertemplate="<b>%{y}</b><br>ISITE: %{x:.1%}<extra></extra>",
    ))
    
    fig.update_layout(
        barmode="overlay",
        title=title,
        xaxis=dict(
            title="% of lab publications",
            tickformat=".0%",
            range=[0, df_fields["share"].max() * 1.15],
        ),
        yaxis=dict(autorange="reversed", title=""),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(l=10, r=10, t=60, b=40),
        height=max(400, len(df_fields) * 25 + 100),
    )
    return fig


def plot_fwci_whiskers(df_fwci: pd.DataFrame, title: str = "") -> go.Figure:
    """Horizontal box-whisker plot for FWCI distribution per field."""
    if df_fwci.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure()
    
    # We'll use shapes for whiskers and boxes
    for i, row in df_fwci.iterrows():
        y = row["field_name"]
        color = row["color"]
        
        # Skip if no data
        if row["count"] <= 0 or pd.isna(row["p50"]):
            continue
        
        # Whisker line (p10 to p90)
        if pd.notna(row["p10"]) and pd.notna(row["p90"]):
            fig.add_trace(go.Scatter(
                x=[row["p10"], row["p90"]],
                y=[y, y],
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
                hoverinfo="skip",
            ))
        
        # Box (p25 to p75) using a bar
        if pd.notna(row["p25"]) and pd.notna(row["p75"]):
            fig.add_trace(go.Bar(
                x=[row["p75"] - row["p25"]],
                y=[y],
                base=row["p25"],
                orientation="h",
                marker=dict(color=color, opacity=0.3),
                width=0.6,
                showlegend=False,
                hovertemplate=f"<b>{y}</b><br>Q1: {row['p25']:.2f}<br>Median: {row['p50']:.2f}<br>Q3: {row['p75']:.2f}<extra></extra>",
            ))
        
        # Median marker
        if pd.notna(row["p50"]):
            fig.add_trace(go.Scatter(
                x=[row["p50"]],
                y=[y],
                mode="markers",
                marker=dict(color=color, size=10, symbol="line-ns", line=dict(width=3, color=color)),
                showlegend=False,
                hovertemplate=f"<b>{y}</b><br>Median: {row['p50']:.2f}<extra></extra>",
            ))
    
    # Add count annotations
    for i, row in df_fwci.iterrows():
        if row["count"] > 0:
            fig.add_annotation(
                x=-0.02,
                y=row["field_name"],
                text=f"{row['count']:,}",
                xref="paper",
                yref="y",
                showarrow=False,
                xanchor="right",
                font=dict(size=10, color="#666"),
            )
    
    xmax = df_fwci["p100"].max() if not df_fwci["p100"].isna().all() else 5
    
    fig.update_layout(
        title=title,
        xaxis=dict(title="FWCI (France)", range=[0, xmax * 1.1]),
        yaxis=dict(autorange="reversed", title=""),
        showlegend=False,
        template="plotly_white",
        margin=dict(l=10, r=10, t=60, b=40),
        height=max(400, len(df_fwci) * 25 + 100),
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
# DOMAIN LEGEND COMPONENT
# ============================================================================

def render_domain_legend():
    """Render inline domain color legend."""
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:16px;">'
        f'<span style="width:14px;height:14px;background:{get_domain_color(d)};border-radius:3px;margin-right:6px;"></span>'
        f'{d}</span>'
        for d in DOMAIN_NAMES_ORDERED
    )
    st.markdown(f'<div style="margin:8px 0 16px 0;">{items}</div>', unsafe_allow_html=True)


# ============================================================================
# PAGE LAYOUT
# ============================================================================

st.title("🏭 Lab View")

# -----------------------------------------------------------------------------
# Section 1: Topline Metrics
# -----------------------------------------------------------------------------

st.subheader("Topline Metrics (2019–2023)")
col1, col2 = st.columns(2)
col1.metric("Number of labs", f"{len(df_labs):,}")
col2.metric("Total publications", f"{df_labs['Pubs total'].sum():,.0f}")

st.divider()

# -----------------------------------------------------------------------------
# Section 2: Per-lab Overview Table
# -----------------------------------------------------------------------------

st.subheader("Per-lab Overview (2019–2023)")

summary = df_labs[[
    "Structure name", "Pole", "Pubs total", "Pubs ISITE (In_LUE)",
    "Pubs international", "Pubs with company",
    "Pubs PPtop10% (subfield)", "Pubs PPtop1% (subfield)", "OpenAlex ID", "ROR"
]].copy()

summary = summary.rename(columns={
    "Structure name": "Lab",
    "Pubs total": "Publications",
    "Pubs ISITE (In_LUE)": "Pubs ISITE",
})

# Compute percentages
summary["% ISITE"] = summary["Pubs ISITE"] / summary["Publications"].replace(0, np.nan) * 100
summary["% international"] = summary["Pubs international"] / summary["Publications"].replace(0, np.nan) * 100
summary["% with company"] = summary["Pubs with company"] / summary["Publications"].replace(0, np.nan) * 100

# Build OpenAlex link
summary["OpenAlex"] = summary["OpenAlex ID"].apply(build_openalex_url)

summary = summary.sort_values("Publications", ascending=False)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
    column_order=[
        "Lab", "Pole", "Publications", "Pubs ISITE", "% ISITE",
        "% international", "% with company",
        "Pubs PPtop10% (subfield)", "Pubs PPtop1% (subfield)", "OpenAlex", "ROR"
    ],
    column_config={
        "Lab": st.column_config.TextColumn("Lab"),
        "Pole": st.column_config.TextColumn("Pole"),
        "Publications": st.column_config.NumberColumn("Publications", format="%.0f"),
        "Pubs ISITE": st.column_config.NumberColumn("Pubs ISITE", format="%.0f"),
        "% ISITE": st.column_config.ProgressColumn("% ISITE", format="%.1f%%", min_value=0, max_value=summary["% ISITE"].max()),
        "% international": st.column_config.ProgressColumn("% international", format="%.1f%%", min_value=0, max_value=summary["% international"].max()),
        "% with company": st.column_config.ProgressColumn("% with company", format="%.1f%%", min_value=0, max_value=summary["% with company"].max()),
        "Pubs PPtop10% (subfield)": st.column_config.NumberColumn("Top 10%", format="%.0f"),
        "Pubs PPtop1% (subfield)": st.column_config.NumberColumn("Top 1%", format="%.0f"),
        "OpenAlex": st.column_config.LinkColumn("OpenAlex"),
        "ROR": st.column_config.TextColumn("ROR"),
    },
)

st.divider()

# -----------------------------------------------------------------------------
# Section 3: Single Lab Analysis
# -----------------------------------------------------------------------------

st.subheader("Lab Analysis")

selected_lab = st.selectbox("Select a lab", lab_names, index=0)
row = df_labs.loc[df_labs["Structure name"] == selected_lab].iloc[0]
pubs_total = safe_int(row.get("Pubs total", 0))

# --- KPI Row ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("Publications (2019–2023)", f"{pubs_total:,}")
k2.metric("… incl. ISITE", f"{safe_int(row.get('Pubs ISITE (In_LUE)', 0)):,}")
k3.metric("… incl. Top 10%", f"{safe_int(row.get('Pubs PPtop10% (subfield)', 0)):,}")
k4.metric("… incl. Top 1%", f"{safe_int(row.get('Pubs PPtop1% (subfield)', 0)):,}")

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
st.markdown("#### Field Distribution (% of lab) — Total & ISITE")
df_fields = build_field_distribution_table(row, pubs_total)
fig_fields = plot_field_distribution(df_fields)
st.plotly_chart(fig_fields, use_container_width=True)

# --- FWCI Whiskers ---
st.markdown("#### FWCI Distribution by Field (vs France)")
df_fwci = build_fwci_whisker_table(row)
fig_fwci = plot_fwci_whiskers(df_fwci)
st.plotly_chart(fig_fwci, use_container_width=True)

st.markdown("---")

# --- Top 5 Internal Partners ---
st.markdown("#### Top 5 Internal Partners")
df_collabs = parse_internal_collabs_blob(row.get("Top 10 internal lab/other collabs (type,count,ratio,FWCI)", ""))
df_collabs = pad_dataframe(df_collabs.head(5), 5, numeric_cols=["count", "ratio", "fwci"])
df_collabs["% of lab pubs"] = df_collabs["ratio"] * 100

st.dataframe(
    df_collabs,
    use_container_width=True,
    hide_index=True,
    column_order=["name", "type", "count", "% of lab pubs", "fwci"],
    column_config={
        "name": st.column_config.TextColumn("Partner"),
        "type": st.column_config.TextColumn("Type"),
        "count": st.column_config.NumberColumn("Co-pubs", format="%.0f"),
        "% of lab pubs": st.column_config.ProgressColumn("% of lab pubs", format="%.1f%%", min_value=0, max_value=df_collabs["% of lab pubs"].max() or 100),
        "fwci": st.column_config.NumberColumn("Avg FWCI", format="%.2f"),
    },
)

st.markdown("---")

# --- Top 10 International Partners ---
st.markdown("#### Top 10 International Partners")
pct_intl = safe_int(row.get("Pubs international", 0)) / max(1, pubs_total) * 100
st.metric("% international publications", f"{pct_intl:.1f}%")

df_intl = build_international_partners_table(row)
df_intl = pad_dataframe(df_intl, 10, numeric_cols=["Co-pubs", "% of UL copubs"])

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
    },
)

st.markdown("---")

# --- Top 10 French Partners ---
st.markdown("#### Top 10 French Partners")
df_fr = build_french_partners_table(row)
df_fr = pad_dataframe(df_fr, 10, numeric_cols=["Co-pubs", "% of UL copubs"])

st.dataframe(
    df_fr,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Partner": st.column_config.TextColumn("Partner"),
        "Co-pubs": st.column_config.NumberColumn("Co-pubs", format="%.0f"),
        "% of UL copubs": st.column_config.ProgressColumn("% of UL copubs", format="%.1f%%", min_value=0, max_value=100),
        "Type": st.column_config.TextColumn("Type"),
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