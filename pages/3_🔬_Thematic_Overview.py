"""
Thematic Overview - High-level exploration of UL's research portfolio.

Sections:
1. Interactive Treemap (Domain → Field → Subfield → Topic)
2. Domains table + FWCI boxplots
3. Fields table + FWCI boxplots
4. Subfields table
5. Topics table (OpenAlex)
6. Research Topics table (Topic Model) + Heatmap
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from lib.helpers import (
    DOMAIN_ORDER,
    DOMAIN_COLORS,
    DOMAIN_EMOJI,
    get_domain_id_to_name,
    get_field_id_to_name,
    get_field_id_to_domain_id,
    get_subfield_id_to_name,
    get_subfield_id_to_domain_id,
    get_field_order_by_domain,
    get_domain_color,
    DOMAIN_NAMES_ORDERED,
    safe_float,
    parse_pipe_float_list,
    render_domain_legend,
)

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="Thematic Overview | UL Bibliometrics",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Thematic Overview")
st.markdown("Explore Université de Lorraine's research portfolio across domains, fields, subfields, and topics.")

# =============================================================================
# Load data
# =============================================================================
@st.cache_data
def load_thematic_overview():
    return pd.read_parquet("data/thematic_overview.parquet")

@st.cache_data
def load_treemap_hierarchy():
    return pd.read_parquet("data/treemap_hierarchy.parquet")

@st.cache_data
def load_tm_labels():
    return pd.read_parquet("data/TM_labels.parquet")

df_overview = load_thematic_overview()
df_treemap_raw = load_treemap_hierarchy()
df_tm_labels = load_tm_labels()

# Lookups
domain_id2name = get_domain_id_to_name()
field_id2name = get_field_id_to_name()
field_id2domain = get_field_id_to_domain_id()
subfield_id2name = get_subfield_id_to_name()
subfield_id2domain = get_subfield_id_to_domain_id()

# =============================================================================
# Helper functions
# =============================================================================
def get_domain_name_from_id(dom_id):
    try:
        return domain_id2name.get(int(dom_id), "Other")
    except (ValueError, TypeError):
        return "Other"

def get_domain_emoji(dom_name):
    return DOMAIN_EMOJI.get(dom_name, "⬜")

def format_pct(val):
    if pd.isna(val):
        return "—"
    return f"{val*100:.1f}%"

def format_cagr(val):
    if pd.isna(val):
        return "—"
    arrow = "↑" if val > 0 else ("↓" if val < 0 else "→")
    return f"{arrow} {val*100:+.1f}%"

def parse_fwci_boxplot(blob):
    if pd.isna(blob) or not str(blob).strip():
        return None
    vals = parse_pipe_float_list(blob)
    if len(vals) < 7:
        return None
    return {"p0": vals[0], "p10": vals[1], "p25": vals[2], "p50": vals[3], 
            "p75": vals[4], "p90": vals[5], "p100": vals[6]}

def parse_pubs_per_domain(blob):
    if pd.isna(blob) or not str(blob).strip():
        return {}
    result = {}
    for part in str(blob).split("|"):
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                result[int(k)] = int(v)
            except ValueError:
                pass
    return result

# =============================================================================
# Section 1: Interactive Treemap
# =============================================================================
st.markdown("---")
st.markdown("## 📊 Research Portfolio Treemap")

st.markdown("""
**How to read this chart**: Each rectangle represents a thematic area. Size reflects publication volume.
Click to drill down from domains → fields → subfields → topics. Use the breadcrumb trail to navigate back.
""")

# Prepare treemap data with additional count columns
df_treemap = df_treemap_raw.copy()
df_treemap["count_top10"] = (df_treemap["pct_top10"] * df_treemap["pubs"]).round().astype(int)
df_treemap["count_top1"] = ((df_treemap["pct_top10"] / 10) * df_treemap["pubs"]).round().astype(int)  # Approximate
df_treemap["count_isite"] = (df_treemap["pct_isite"] * df_treemap["pubs"]).round().astype(int)

# Color metric selector
color_metric = st.selectbox(
    "Color by:",
    ["fwci_median", "count_isite", "count_top10", "pct_international"],
    format_func=lambda x: {
        "fwci_median": "Median FWCI (citation impact)",
        "count_isite": "Count of ISITE publications",
        "count_top10": "Count of Top 10% publications",
        "pct_international": "% International collaborations",
    }.get(x, x)
)

# Build treemap with custom color scale for FWCI
if color_metric == "fwci_median":
    # Custom scale: red (0) -> grey (1) -> green (2+)
    fig_treemap = px.treemap(
        df_treemap,
        ids="id",
        names="name",
        parents="parent_id",
        values="pubs",
        color="fwci_median",
        color_continuous_scale=[
            [0.0, "#d73027"],    # Red for 0
            [0.5, "#999999"],    # Grey for 1
            [1.0, "#1a9850"],    # Green for 2+
        ],
        range_color=[0, 2],
    )
else:
    fig_treemap = px.treemap(
        df_treemap,
        ids="id",
        names="name",
        parents="parent_id",
        values="pubs",
        color=color_metric,
        color_continuous_scale="Blues",
    )

# Update hover template
fig_treemap.update_traces(
    customdata=np.stack([
        df_treemap["pubs"],
        df_treemap["fwci_median"],
        df_treemap["pct_top10"] * 100,
        df_treemap["pct_international"] * 100,
        df_treemap["pct_isite"] * 100,
        df_treemap["count_isite"],
        df_treemap["count_top10"],
    ], axis=-1),
    hovertemplate="<b>%{label}</b><br>" +
                  "Publications: %{customdata[0]:,}<br>" +
                  "FWCI median: %{customdata[1]:.2f}<br>" +
                  "Top 10%%: %{customdata[2]:.1f}%<br>" +
                  "International: %{customdata[3]:.1f}%<br>" +
                  "ISITE: %{customdata[4]:.1f}%<br>" +
                  "ISITE count: %{customdata[5]:,}<br>" +
                  "Top 10%% count: %{customdata[6]:,}<extra></extra>"
)

fig_treemap.update_layout(
    margin=dict(t=30, l=10, r=10, b=10),
    height=600,
)

st.plotly_chart(fig_treemap, use_container_width=True)

# =============================================================================
# Section 2: Domains
# =============================================================================
st.markdown("---")
st.markdown("## 🌐 Domains")
render_domain_legend()

df_domains = df_overview[df_overview["level"] == "domain"].copy()
df_domains["domain_id"] = df_domains["id"].astype(int)
df_domains = df_domains.sort_values("domain_id", key=lambda x: x.map({d: i for i, d in enumerate(DOMAIN_ORDER)}))

st.markdown("### Overview by Domain")

# Build display table
domain_table = []
for _, row in df_domains.iterrows():
    dom_name = row["name"]
    domain_table.append({
        "Domain": f"{get_domain_emoji(dom_name)} {dom_name}",
        "Pubs": int(row["pubs_total"]),
        "% Total": format_pct(row["pubs_pct_of_ul"]),
        "% ISITE": row["pct_isite"],  # Keep as float for progress bar
        "% Top 10%": format_pct(row["pct_top10"]),
        "% Top 1%": format_pct(row["pct_top1"]),
        "% Int'l": format_pct(row["pct_international"]),
        "% Company": format_pct(row["pct_company"]),
        "% SDG": format_pct(row["pct_sdg"]),
        "CAGR": format_cagr(row["cagr_2019_2023"]),
    })

df_domain_display = pd.DataFrame(domain_table)
st.dataframe(
    df_domain_display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "% ISITE": st.column_config.ProgressColumn(
            "% ISITE",
            min_value=0,
            max_value=1,
            format="%.1f%%",
        ),
    }
)

# FWCI Distribution boxplots - horizontal, separated
st.markdown("### FWCI Distribution by Domain")

use_extreme = st.toggle("Include extreme values (p0, p100)", value=False, key="domain_extreme")

boxplot_data = []
for _, row in df_domains.iterrows():
    bp = parse_fwci_boxplot(row["fwci_boxplot"])
    if bp:
        dom_name = row["name"]
        boxplot_data.append({
            "domain": dom_name,
            "domain_id": row["domain_id"],
            "color": DOMAIN_COLORS.get(dom_name, "#7f7f7f"),
            "count": int(row["pubs_total"]),
            **bp
        })

if boxplot_data:
    # Sort by domain order
    boxplot_data = sorted(boxplot_data, key=lambda x: DOMAIN_ORDER.index(x["domain_id"]) if x["domain_id"] in DOMAIN_ORDER else 99)
    
    fig_box = go.Figure()
    
    for i, item in enumerate(boxplot_data):
        if use_extreme:
            lower, upper = item["p0"], item["p100"]
        else:
            lower, upper = item["p10"], item["p90"]
        
        fig_box.add_trace(go.Box(
            x=[item["domain"]],
            lowerfence=[lower],
            q1=[item["p25"]],
            median=[item["p50"]],
            q3=[item["p75"]],
            upperfence=[upper],
            marker_color=item["color"],
            fillcolor=item["color"],
            line=dict(color=item["color"]),
            boxpoints=False,
            name=item["domain"],
            showlegend=False,
        ))
        
        # Add count annotation below
        fig_box.add_annotation(
            x=item["domain"],
            y=-0.15,
            yref="paper",
            text=f"n={item['count']:,}",
            showarrow=False,
            font=dict(size=11, color="#666"),
        )
    
    fig_box.update_layout(
        height=350,
        margin=dict(t=30, l=50, r=30, b=60),
        yaxis_title="FWCI",
        xaxis_title="",
    )
    st.plotly_chart(fig_box, use_container_width=True)

# =============================================================================
# Section 3: Fields
# =============================================================================
st.markdown("---")
st.markdown("## 📚 Fields")
render_domain_legend()

df_fields = df_overview[df_overview["level"] == "field"].copy()
df_fields["field_id"] = df_fields["id"].astype(int)
df_fields["domain_id"] = df_fields["parent_id"].astype(int)
df_fields["domain_name"] = df_fields["domain_id"].map(domain_id2name)

# Sort by pubs descending for table
df_fields_table = df_fields.sort_values("pubs_total", ascending=False)

st.markdown("### Overview by Field")

field_table = []
for _, row in df_fields_table.iterrows():
    dom_name = row["domain_name"]
    field_table.append({
        "": get_domain_emoji(dom_name),
        "Field": row["name"],
        "Pubs": int(row["pubs_total"]),
        "% Total": format_pct(row["pubs_pct_of_ul"]),
        "% ISITE": row["pct_isite"],  # Keep as float for progress bar
        "% Top 10%": format_pct(row["pct_top10"]),
        "% Top 1%": format_pct(row["pct_top1"]),
        "% Int'l": format_pct(row["pct_international"]),
        "% Company": format_pct(row["pct_company"]),
        "CAGR": format_cagr(row["cagr_2019_2023"]),
    })

df_field_display = pd.DataFrame(field_table)
st.dataframe(
    df_field_display,
    use_container_width=True,
    hide_index=True,
    height=500,
    column_config={
        "% ISITE": st.column_config.ProgressColumn(
            "% ISITE",
            min_value=0,
            max_value=1,
            format="%.1f%%",
        ),
    }
)

# FWCI Distribution boxplots - horizontal, ordered by domain
st.markdown("### FWCI Distribution by Field")

use_extreme_fields = st.toggle("Include extreme values (p0, p100)", value=False, key="field_extreme")

# Sort by domain order for boxplot
field_order = get_field_order_by_domain()
df_fields_sorted = df_fields.copy()
df_fields_sorted["sort_order"] = df_fields_sorted["field_id"].map({fid: i for i, fid in enumerate(field_order)})
df_fields_sorted = df_fields_sorted.sort_values("sort_order")

boxplot_data_fields = []
for _, row in df_fields_sorted.iterrows():
    bp = parse_fwci_boxplot(row["fwci_boxplot"])
    if bp and row["pubs_total"] > 0:
        field_id = row["field_id"]
        dom_id = field_id2domain.get(field_id, 0)
        dom_name = domain_id2name.get(dom_id, "Other")
        boxplot_data_fields.append({
            "field": row["name"],
            "field_id": field_id,
            "color": DOMAIN_COLORS.get(dom_name, "#7f7f7f"),
            "count": int(row["pubs_total"]),
            **bp
        })

if boxplot_data_fields:
    fig_box_fields = go.Figure()
    
    for item in boxplot_data_fields:
        if use_extreme_fields:
            lower, upper = item["p0"], item["p100"]
        else:
            lower, upper = item["p10"], item["p90"]
        
        fig_box_fields.add_trace(go.Box(
            x=[item["field"]],
            lowerfence=[lower],
            q1=[item["p25"]],
            median=[item["p50"]],
            q3=[item["p75"]],
            upperfence=[upper],
            marker_color=item["color"],
            fillcolor=item["color"],
            line=dict(color=item["color"]),
            boxpoints=False,
            name=item["field"],
            showlegend=False,
        ))
    
    # Add count annotations
    for i, item in enumerate(boxplot_data_fields):
        fig_box_fields.add_annotation(
            x=item["field"],
            y=-0.12,
            yref="paper",
            text=f"n={item['count']:,}",
            showarrow=False,
            font=dict(size=9, color="#666"),
            textangle=-45,
        )
    
    fig_box_fields.update_layout(
        height=450,
        margin=dict(t=30, l=50, r=30, b=120),
        yaxis_title="FWCI",
        xaxis_title="",
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig_box_fields, use_container_width=True)

# =============================================================================
# Section 4: Subfields
# =============================================================================
st.markdown("---")
st.markdown("## 📖 Subfields")
render_domain_legend()

df_subfields = df_overview[df_overview["level"] == "subfield"].copy()
df_subfields["subfield_id"] = df_subfields["id"].astype(int)
df_subfields["field_id"] = df_subfields["parent_id"].astype(int)
df_subfields["field_name"] = df_subfields["field_id"].map(field_id2name)
df_subfields["domain_id"] = df_subfields["subfield_id"].map(subfield_id2domain)
df_subfields["domain_name"] = df_subfields["domain_id"].map(domain_id2name)

# Filter and search
col_filter1, col_filter2 = st.columns(2)
with col_filter1:
    domain_filter = st.multiselect(
        "Filter by domain:",
        options=list(domain_id2name.values()),
        default=[],
        key="subfield_domain_filter"
    )
with col_filter2:
    search_subfield = st.text_input("Search subfield:", "", key="subfield_search")

df_subfields_filtered = df_subfields.copy()
if domain_filter:
    df_subfields_filtered = df_subfields_filtered[df_subfields_filtered["domain_name"].isin(domain_filter)]
if search_subfield:
    df_subfields_filtered = df_subfields_filtered[
        df_subfields_filtered["name"].str.lower().str.contains(search_subfield.lower(), na=False)
    ]

# Sort by pubs descending
df_subfields_filtered = df_subfields_filtered.sort_values("pubs_total", ascending=False)

subfield_table = []
for _, row in df_subfields_filtered.iterrows():
    dom_name = row["domain_name"] if pd.notna(row["domain_name"]) else "Other"
    subfield_table.append({
        "": get_domain_emoji(dom_name),
        "Subfield": row["name"],
        "Field": row["field_name"] if pd.notna(row["field_name"]) else "",
        "Pubs": int(row["pubs_total"]),
        "% Total": format_pct(row["pubs_pct_of_ul"]),
        "% ISITE": row["pct_isite"],
        "% Top 10%": format_pct(row["pct_top10"]),
        "% Top 1%": format_pct(row["pct_top1"]),
        "% Int'l": format_pct(row["pct_international"]),
        "CAGR": format_cagr(row["cagr_2019_2023"]),
    })

df_subfield_display = pd.DataFrame(subfield_table)
st.dataframe(
    df_subfield_display,
    use_container_width=True,
    hide_index=True,
    height=400,
    column_config={
        "% ISITE": st.column_config.ProgressColumn(
            "% ISITE",
            min_value=0,
            max_value=1,
            format="%.1f%%",
        ),
    }
)
st.caption(f"Showing {len(subfield_table)} subfields")

# =============================================================================
# Section 5: Topics (OpenAlex)
# =============================================================================
st.markdown("---")
st.markdown("## 🏷️ Topics (OpenAlex)")
render_domain_legend()

df_topics = df_overview[df_overview["level"] == "topic"].copy()
df_topics["topic_id"] = df_topics["id"]
df_topics["subfield_id"] = pd.to_numeric(df_topics["parent_id"], errors="coerce").astype("Int64")
df_topics["subfield_name"] = df_topics["subfield_id"].map(subfield_id2name)
df_topics["domain_id"] = df_topics["subfield_id"].map(subfield_id2domain)
df_topics["domain_name"] = df_topics["domain_id"].map(domain_id2name)

# Filter and search
col_filter1, col_filter2 = st.columns(2)
with col_filter1:
    domain_filter_topics = st.multiselect(
        "Filter by domain:",
        options=list(domain_id2name.values()),
        default=[],
        key="topic_domain_filter"
    )
with col_filter2:
    search_topic = st.text_input("Search topic:", "", key="topic_search")

df_topics_filtered = df_topics.copy()
if domain_filter_topics:
    df_topics_filtered = df_topics_filtered[df_topics_filtered["domain_name"].isin(domain_filter_topics)]
if search_topic:
    df_topics_filtered = df_topics_filtered[
        df_topics_filtered["name"].str.lower().str.contains(search_topic.lower(), na=False)
    ]

# Sort by pubs descending, limit display
df_topics_filtered = df_topics_filtered.sort_values("pubs_total", ascending=False).head(200)

topic_table = []
for _, row in df_topics_filtered.iterrows():
    dom_name = row["domain_name"] if pd.notna(row["domain_name"]) else "Other"
    topic_table.append({
        "": get_domain_emoji(dom_name),
        "Topic": row["name"],
        "Subfield": row["subfield_name"] if pd.notna(row["subfield_name"]) else "",
        "Pubs": int(row["pubs_total"]),
        "% Total": format_pct(row["pubs_pct_of_ul"]),
        "% ISITE": row["pct_isite"],
        "% Top 10%": format_pct(row["pct_top10"]),
        "% Int'l": format_pct(row["pct_international"]),
        "CAGR": format_cagr(row["cagr_2019_2023"]),
    })

df_topic_display = pd.DataFrame(topic_table)
st.dataframe(
    df_topic_display,
    use_container_width=True,
    hide_index=True,
    height=400,
    column_config={
        "% ISITE": st.column_config.ProgressColumn(
            "% ISITE",
            min_value=0,
            max_value=1,
            format="%.1f%%",
        ),
    }
)
st.caption(f"Showing top {len(topic_table)} topics by volume")

# =============================================================================
# Section 6: Research Topics (Topic Model)
# =============================================================================
st.markdown("---")
st.markdown("## 🧬 Research Topics (Topic Model)")

st.markdown("""
These topics were identified through bottom-up clustering of research abstracts using LLM-based extraction.
They reveal thematic patterns that may cut across traditional disciplinary boundaries.
""")

df_research = df_overview[df_overview["level"] == "research_topic"].copy()
df_research["rt_id"] = df_research["id"].astype(int)
df_research = df_research.sort_values("pubs_total", ascending=False)

# Table
st.markdown("### Research Topics Overview")

rt_table = []
for _, row in df_research.iterrows():
    rt_table.append({
        "ID": row["rt_id"],
        "Topic": row["name"],
        "Pubs": int(row["pubs_total"]),
        "% Total": format_pct(row["pubs_pct_of_ul"]),
        "% ISITE": row["pct_isite"],
        "% Top 10%": format_pct(row["pct_top10"]),
        "% Top 1%": format_pct(row["pct_top1"]),
        "% Int'l": format_pct(row["pct_international"]),
        "% Company": format_pct(row["pct_company"]),
        "CAGR": format_cagr(row["cagr_2019_2023"]),
    })

df_rt_display = pd.DataFrame(rt_table)
st.dataframe(
    df_rt_display,
    use_container_width=True,
    hide_index=True,
    height=400,
    column_config={
        "% ISITE": st.column_config.ProgressColumn(
            "% ISITE",
            min_value=0,
            max_value=1,
            format="%.1f%%",
        ),
    }
)

# Heatmap: Research Topics x Domains
st.markdown("### Research Topics × Domains Heatmap")

st.markdown("""
This heatmap shows how each research topic distributes across the four scientific domains.
Darker cells indicate higher publication counts. Topics spanning multiple domains reveal interdisciplinary research.
""")

# Build heatmap matrix
heatmap_data = []
for _, row in df_research.iterrows():
    dom_counts = parse_pubs_per_domain(row["pubs_per_domain"])
    heatmap_data.append({
        "Research Topic": row["name"],
        "rt_id": row["rt_id"],
        **{domain_id2name.get(d, f"Domain {d}"): dom_counts.get(d, 0) for d in DOMAIN_ORDER}
    })

df_heatmap = pd.DataFrame(heatmap_data)

# Sort by total pubs
df_heatmap["total"] = df_heatmap[[domain_id2name.get(d) for d in DOMAIN_ORDER]].sum(axis=1)
df_heatmap = df_heatmap.sort_values("total", ascending=True).tail(30)  # Top 30 by volume

# Normalize option
normalize = st.checkbox("Normalize by row (show domain distribution per topic)", value=False)

domain_cols = [domain_id2name.get(d) for d in DOMAIN_ORDER]
z_values = df_heatmap[domain_cols].values.astype(float)

if normalize:
    row_sums = z_values.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    z_values_normalized = z_values / row_sums
    
    # Custom hover text for normalized values
    hover_text = []
    for i, row_label in enumerate(df_heatmap["Research Topic"].tolist()):
        row_hover = []
        for j, col in enumerate(domain_cols):
            pct = z_values_normalized[i, j] * 100
            count = int(z_values[i, j])
            row_hover.append(f"{row_label}<br>{col}: {pct:.1f}% ({count} pubs)")
        hover_text.append(row_hover)
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=z_values_normalized,
        x=domain_cols,
        y=df_heatmap["Research Topic"].tolist(),
        colorscale="Blues",
        hoverinfo="text",
        text=hover_text,
        zmin=0,
        zmax=1,
        colorbar=dict(tickformat=".0%"),
    ))
else:
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=z_values,
        x=domain_cols,
        y=df_heatmap["Research Topic"].tolist(),
        colorscale="Blues",
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:,} pubs<extra></extra>",
    ))

fig_heatmap.update_layout(
    height=max(500, len(df_heatmap) * 25),
    margin=dict(t=30, l=400, r=30, b=50),
    xaxis_title="Domain",
    yaxis_title="",
    yaxis=dict(
        tickmode="array",
        tickvals=list(range(len(df_heatmap))),
        ticktext=df_heatmap["Research Topic"].tolist(),
        automargin=True,
    ),
)

st.plotly_chart(fig_heatmap, use_container_width=True)

# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.caption("Data: Université de Lorraine publications 2019-2023 | OpenAlex + custom topic modeling")