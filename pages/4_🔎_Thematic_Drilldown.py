"""
Thematic Drill-Down - Detailed exploration of a specific domain, field, subfield, or research topic.
FIXED VERSION: Removed unnecessary load_lab_info and load_partners_base dependencies.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

from lib.helpers import (
    get_domain_id_to_name,
    get_field_id_to_name,
    get_field_id_to_domain_id,
    get_subfield_id_to_name,
    get_subfield_id_to_domain_id,
    safe_float,
    safe_int,
)

from lib.data_cache import (
    load_thematic_overview,
    load_thematic_sublevels,
    load_thematic_contributions,
    load_thematic_partners,
    load_thematic_authors,
    load_tm_labels,
)

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="Thematic Drill-Down | UL Bibliometrics",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Thematic Drill-Down")
st.markdown("Explore detailed metrics for a specific domain, field, subfield, or research topic.")

# =============================================================================
# Constants
# =============================================================================
LEVEL_LABELS = {
    "domain": "Domain",
    "field": "Field",
    "subfield": "Subfield",
    "research_topic": "Research Topic",
}

CHILD_LEVEL_LABELS = {
    "domain": "Field",
    "field": "Subfield",
    "subfield": "Topic",
}

STRUCTURE_TYPE_COLORS = {
    "lab": "#4e79a7",
    "experimental": "#f28e2b",
    "other": "#76b7b2",
}

# =============================================================================
# Load data (SIMPLIFIED - removed unnecessary loads)
# =============================================================================
df_overview = load_thematic_overview()
df_sublevels = load_thematic_sublevels()
df_contributions = load_thematic_contributions()
df_partners = load_thematic_partners()
df_authors = load_thematic_authors()
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
def format_pct(val):
    if pd.isna(val):
        return "—"
    return f"{val*100:.1f}%"

def format_cagr(val):
    if pd.isna(val):
        return "—"
    arrow = "↑" if val > 0 else ("↓" if val < 0 else "→")
    return f"{arrow} {val*100:+.1f}%"

def parse_year_counts(blob):
    """Parse '2019:120|2020:135|...' into dict {year: count}."""
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

def parse_top_items(blob, expected_fields):
    """
    Parse pipe-separated items with colon-separated fields.
    FIXED: Now handles fields with varying counts more robustly.
    """
    if pd.isna(blob) or not str(blob).strip():
        return []
    results = []
    for item in str(blob).split("|"):
        parts = item.split(":")
        if len(parts) >= len(expected_fields):
            row = {field: parts[i] for i, field in enumerate(expected_fields)}
            results.append(row)
    return results

def add_spaces_to_name(name):
    """Add spaces before capital letters in compressed names."""
    if not name:
        return name
    spaced = re.sub(r'([a-zàâäéèêëïîôùûüç])([A-ZÀÂÄÉÈÊËÏÎÔÙÛÜÇ])', r'\1 \2', str(name))
    return spaced

def get_element_options(level):
    """Get available elements for a given level."""
    df_level = df_overview[df_overview["level"] == level].copy()
    df_level = df_level.sort_values("pubs_total", ascending=False)
    options = []
    for _, row in df_level.iterrows():
        label = f"{row['name']} ({int(row['pubs_total']):,} pubs)"
        options.append((row["id"], label))
    return options

def get_element_data(level, element_id):
    """Get overview data for a specific element."""
    mask = (df_overview["level"] == level) & (df_overview["id"] == str(element_id))
    rows = df_overview[mask]
    if rows.empty:
        return None
    return rows.iloc[0]

def get_sublevel_data(parent_level, parent_id):
    """Get sublevel breakdown data."""
    mask = (df_sublevels["parent_level"] == parent_level) & (df_sublevels["parent_id"] == str(parent_id))
    return df_sublevels[mask].copy()

def get_contribution_data(level, element_id):
    """Get contribution data (research topics, depts, labs)."""
    mask = (df_contributions["level"] == level) & (df_contributions["id"] == str(element_id))
    rows = df_contributions[mask]
    if rows.empty:
        return None
    return rows.iloc[0]

def find_column(row_or_df, pattern):
    """Find column name containing a pattern (handles columns with format descriptions)."""
    if hasattr(row_or_df, 'index'):
        # It's a Series (row)
        cols = row_or_df.index
    else:
        # It's a DataFrame
        cols = row_or_df.columns
    
    for col in cols:
        if pattern in col:
            return col
    return None

def get_partner_data(level, element_id):
    """Get partner data."""
    mask = (df_partners["level"] == level) & (df_partners["id"] == str(element_id))
    rows = df_partners[mask]
    if rows.empty:
        return None
    return rows.iloc[0]

def get_author_data(level, element_id):
    """Get author data."""
    mask = (df_authors["level"] == level) & (df_authors["id"] == str(element_id))
    rows = df_authors[mask]
    if rows.empty:
        return None
    return rows.iloc[0]

def render_structure_type_legend():
    """Render legend for structure types."""
    items = ""
    for stype, color in STRUCTURE_TYPE_COLORS.items():
        items += (
            f'<span style="display:inline-flex;align-items:center;margin-right:16px;">'
            f'<span style="width:14px;height:14px;background:{color};border-radius:3px;margin-right:6px;"></span>'
            f'{stype.title()}</span>'
        )
    st.markdown(f'<div style="margin:8px 0 16px 0;">{items}</div>', unsafe_allow_html=True)

def get_research_topic_keywords(topic_id):
    """Get keywords for a research topic from TM_labels."""
    if df_tm_labels.empty:
        return []
    
    try:
        topic_id_int = int(topic_id)
    except (ValueError, TypeError):
        return []
    
    rt_labels = df_tm_labels[df_tm_labels["dimension"] == "research topic"]
    matching = rt_labels[rt_labels["topic_id"] == topic_id_int]
    
    if matching.empty:
        return []
    
    keywords_str = matching.iloc[0].get("keywords", "")
    if pd.isna(keywords_str) or not keywords_str:
        return []
    
    return [kw.strip() for kw in str(keywords_str).split("|") if kw.strip()]

def render_keywords_badges(keywords):
    """Render keywords as styled badges."""
    if not keywords:
        return
    
    badges_html = ""
    for kw in keywords:
        badges_html += (
            f'<span style="display:inline-block;background:#e8f4f8;color:#0c5460;'
            f'padding:4px 12px;margin:4px;border-radius:16px;font-size:0.9em;'
            f'border:1px solid #bee5eb;">{kw}</span>'
        )
    st.markdown(f'<div style="margin:12px 0;">{badges_html}</div>', unsafe_allow_html=True)

# =============================================================================
# Section 1: Selector
# =============================================================================
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    level = st.selectbox(
        "Select level:",
        ["domain", "field", "subfield", "research_topic"],
        format_func=lambda x: {
            "domain": "🌐 Domain",
            "field": "📚 Field",
            "subfield": "📖 Subfield",
            "research_topic": "🧬 Research Topic (Topic Model)",
        }.get(x, x)
    )

with col2:
    element_options = get_element_options(level)
    if element_options:
        element_id = st.selectbox(
            "Select element:",
            options=[opt[0] for opt in element_options],
            format_func=lambda x: dict(element_options).get(x, x)
        )
    else:
        st.warning("No elements found for this level.")
        st.stop()

# Get element data
element_data = get_element_data(level, element_id)
if element_data is None:
    st.error("Element data not found.")
    st.stop()

element_name = element_data['name']
level_label = LEVEL_LABELS.get(level, level.title())

# Display element name as header
st.markdown(f"## {element_name}")

# For research topics, show methodology info and keywords
if level == "research_topic":
    st.markdown("""
    <div style="background:#f8f9fa;padding:12px 16px;border-radius:8px;border-left:4px solid #6c757d;margin-bottom:16px;">
    <strong>📌 About this topic:</strong> Research topics are identified through a bottom-up approach: 
    key themes are extracted from publication abstracts using a Large Language Model, 
    then grouped into coherent clusters using k-means clustering.
    </div>
    """, unsafe_allow_html=True)
    
    keywords = get_research_topic_keywords(element_id)
    if keywords:
        st.markdown("**Keywords:**")
        render_keywords_badges(keywords)

# =============================================================================
# Section 2: Topline KPIs
# =============================================================================
st.markdown("---")

st.markdown("#### 📊 Volume & Growth")
kpi_cols1 = st.columns(4)
with kpi_cols1[0]:
    st.metric("Publications", f"{int(element_data['pubs_total']):,}")
with kpi_cols1[1]:
    st.metric("% of UL Total", format_pct(element_data['pubs_pct_of_ul']))
with kpi_cols1[2]:
    st.metric("CAGR 2019-23", format_cagr(element_data['cagr_2019_2023']))

st.markdown("#### 🎯 Citation Impact")
kpi_cols2 = st.columns(4)
with kpi_cols2[0]:
    fwci_median = element_data['fwci_median']
    st.metric("Median FWCI", f"{fwci_median:.2f}" if pd.notna(fwci_median) else "—")
with kpi_cols2[1]:
    fwci_mean = element_data['fwci_mean']
    st.metric("Avg. FWCI", f"{fwci_mean:.2f}" if pd.notna(fwci_mean) else "—")
with kpi_cols2[2]:
    st.metric("% Top 10%", format_pct(element_data['pct_top10']))
with kpi_cols2[3]:
    st.metric("% Top 1%", format_pct(element_data['pct_top1']))

st.markdown("#### 🤝 Collaborations")
kpi_cols3 = st.columns(4)
with kpi_cols3[0]:
    st.metric("🌍 % International", format_pct(element_data['pct_international']))
with kpi_cols3[1]:
    st.metric("🏢 % Company", format_pct(element_data['pct_company']))

st.markdown("#### 🌱 Challenge-oriented")
kpi_cols4 = st.columns(4)
with kpi_cols4[0]:
    st.metric("% SDG-related", format_pct(element_data['pct_sdg']))
with kpi_cols4[1]:
    st.metric("% ISITE", format_pct(element_data['pct_isite']))

# =============================================================================
# Section 3: Sublevel Breakdown (for OA taxonomy only)
# =============================================================================
if level in ["domain", "field", "subfield"]:
    st.markdown("---")
    
    child_level_label = CHILD_LEVEL_LABELS.get(level, "Sub-element")
    st.markdown(f"### 📊 {child_level_label} mix within {element_name}")
    
    df_sub = get_sublevel_data(level, element_id)
    
    if not df_sub.empty:
        df_sub = df_sub.sort_values("pubs_total", ascending=False)
        
        sub_table = []
        for _, row in df_sub.iterrows():
            sub_table.append({
                "Name": row["child_name"],
                "Pubs": int(row["pubs_total"]),
                f"{level_label} share": row["pubs_pct_of_parent"] * 100,
                "ISITE contribution": row["pct_isite"] * 100,
                "% Top 10%": format_pct(row["pct_top10"]),
                "% Top 1%": format_pct(row["pct_top1"]),
                "% International": format_pct(row["pct_international"]),
                "Median FWCI": f"{row['fwci_median']:.2f}" if pd.notna(row['fwci_median']) else "—",
                "Avg. FWCI": f"{row['fwci_mean']:.2f}" if pd.notna(row['fwci_mean']) else "—",
                "CAGR": format_cagr(row["cagr_2019_2023"]),
            })
        
        df_sub_display = pd.DataFrame(sub_table)
        st.dataframe(
            df_sub_display,
            use_container_width=True,
            hide_index=True,
            height=min(400, 35 + len(sub_table) * 35),
            column_config={
                f"{level_label} share": st.column_config.ProgressColumn(
                    f"{level_label} share",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                ),
                "ISITE contribution": st.column_config.ProgressColumn(
                    "ISITE contribution",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                ),
            }
        )
        
        # Time Evolution Charts
        st.markdown(f"### 📈 Time Evolution of {child_level_label}s")
        
        time_data = []
        for _, row in df_sub.iterrows():
            year_counts = parse_year_counts(row["pubs_per_year"])
            for year, count in year_counts.items():
                time_data.append({
                    "Year": year,
                    "Name": row["child_name"],
                    "Count": count,
                })
        
        df_time = pd.DataFrame(time_data)
        
        if not df_time.empty:
            top_names = df_sub.nlargest(10, "pubs_total")["child_name"].tolist()
            df_time_top = df_time[df_time["Name"].isin(top_names)]
            
            df_time_other = df_time[~df_time["Name"].isin(top_names)].groupby("Year")["Count"].sum().reset_index()
            df_time_other["Name"] = "Other"
            
            df_time_plot = pd.concat([df_time_top, df_time_other], ignore_index=True)
            
            all_names = top_names + ["Other"]
            color_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set2
            color_map = {name: color_palette[i % len(color_palette)] for i, name in enumerate(all_names)}
            
            st.markdown("**Absolute values**")
            fig_abs = px.line(
                df_time_plot,
                x="Year",
                y="Count",
                color="Name",
                color_discrete_map=color_map,
                markers=True,
            )
            fig_abs.update_layout(
                height=400,
                margin=dict(t=30, l=50, r=30, b=50),
                xaxis=dict(
                    dtick=1,
                    showgrid=True,
                    gridcolor="lightgrey",
                    gridwidth=0.5,
                ),
                yaxis_title="Publications",
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_abs, use_container_width=True)
            
            st.markdown("**Relative share (100% stacked)**")
            df_time_pct = df_time_plot.copy()
            year_totals = df_time_pct.groupby("Year")["Count"].transform("sum")
            df_time_pct["Share"] = (df_time_pct["Count"] / year_totals * 100).fillna(0)
            
            fig_stack = px.area(
                df_time_pct,
                x="Year",
                y="Share",
                color="Name",
                color_discrete_map=color_map,
                groupnorm="percent",
            )
            fig_stack.update_traces(
                hovertemplate="Year=%{x}<br>Share=%{y:.2f}%<extra>%{fullData.name}</extra>"
            )
            fig_stack.update_layout(
                height=400,
                margin=dict(t=30, l=50, r=30, b=50),
                xaxis=dict(
                    dtick=1,
                    showgrid=True,
                    gridcolor="lightgrey",
                    gridwidth=0.5,
                ),
                yaxis=dict(title="Share (%)", range=[0, 100]),
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_stack, use_container_width=True)

# =============================================================================
# Section 5: Contribution Analysis
# =============================================================================
st.markdown("---")
st.markdown("### 🏗️ Contribution Analysis")

contrib_data = get_contribution_data(level, element_id)

if contrib_data is not None:
    if level != "research_topic":
        st.markdown("**Top 10 contributing Research Topics**")
        # Find the actual column name (may include format description)
        rt_col = find_column(contrib_data, "top_research_topics")
        rt_items = []
        if rt_col:
            rt_items = parse_top_items(
                contrib_data.get(rt_col, ""),
                ["id", "label", "count", "pct"]
            )
        if rt_items:
            rt_df = pd.DataFrame(rt_items)
            rt_df["count"] = rt_df["count"].apply(safe_int)
            rt_df["pct"] = rt_df["pct"].apply(safe_float)
            
            fig_rt = go.Figure(go.Bar(
                y=rt_df["label"].tolist()[::-1],
                x=rt_df["count"].tolist()[::-1],
                orientation="h",
                marker_color="#4e79a7",
                text=[f"{p*100:.1f}%" for p in rt_df["pct"].tolist()[::-1]],
                textposition="auto",
            ))
            fig_rt.update_layout(
                height=450,
                margin=dict(t=10, l=10, r=10, b=10),
                xaxis_title="Publications",
                yaxis_title="",
            )

            fig_rt.update_yaxes(tickfont_size=14)

            st.plotly_chart(fig_rt, use_container_width=True)
        else:
            st.info("No research topic data.")
    
    st.markdown("**Contributing departments**")
    dept_col = find_column(contrib_data, "department_breakdown")
    dept_items = []
    if dept_col:
        dept_items = parse_top_items(
            contrib_data.get(dept_col, ""),
            ["dept", "count", "pct"]
        )
    if dept_items:
        dept_df = pd.DataFrame(dept_items)
        dept_df["count"] = dept_df["count"].apply(safe_int)
        dept_df["pct"] = dept_df["pct"].apply(safe_float)
        dept_df = dept_df.sort_values("count", ascending=True)
        
        fig_dept = go.Figure(go.Bar(
            y=dept_df["dept"].tolist(),
            x=dept_df["count"].tolist(),
            orientation="h",
            marker_color="#59a14f",
            text=[f"{p*100:.1f}%" for p in dept_df["pct"].tolist()],
            textposition="auto",
            insidetextfont=dict(color="white"),
            outsidetextfont=dict(color="black"),
        ))
        fig_dept.update_layout(
            height=max(200, len(dept_df) * 40),
            margin=dict(t=10, l=10, r=10, b=10),
            xaxis_title="Publications",
            yaxis_title="",
        )
        st.plotly_chart(fig_dept, use_container_width=True)
    else:
        st.info("No department data.")
    
    st.markdown("**Top 10 contributing labs / internal structures**")
    render_structure_type_legend()
    
    lab_col = find_column(contrib_data, "top_labs")
    lab_items = []
    if lab_col:
        lab_items = parse_top_items(
            contrib_data.get(lab_col, ""),
            ["ror", "name", "type", "count", "pct"]
        )
    if lab_items:
        lab_df = pd.DataFrame(lab_items)
        lab_df["count"] = lab_df["count"].apply(safe_int)
        lab_df["pct"] = lab_df["pct"].apply(safe_float)
        
        lab_df["color"] = lab_df["type"].apply(
            lambda x: STRUCTURE_TYPE_COLORS.get(x, STRUCTURE_TYPE_COLORS["other"])
        )
        
        lab_df = lab_df.sort_values("count", ascending=True).tail(10)
        
        fig_lab = go.Figure(go.Bar(
            y=lab_df["name"].tolist(),
            x=lab_df["count"].tolist(),
            orientation="h",
            marker_color=lab_df["color"].tolist(),
            text=[f"{p*100:.1f}%" for p in lab_df["pct"].tolist()],
            textposition="auto",
        ))
        fig_lab.update_layout(
            height=350,
            margin=dict(t=10, l=10, r=10, b=10),
            xaxis_title="Publications",
            yaxis_title="",
        )
        st.plotly_chart(fig_lab, use_container_width=True)
    else:
        st.info("No lab data.")

# =============================================================================
# Section 6: Partner Tables
# =============================================================================
st.markdown("---")
st.markdown("### 🤝 Top Partners")

partner_data = get_partner_data(level, element_id)

if partner_data is not None:
    st.markdown("**Top 20 International Partners**")
    int_col = [c for c in df_partners.columns if "top_int_partners" in c][0]
    # Format: id:name:country:type:copubs:share_ul:share_int:share_partner:fwci (9 fields)
    int_items = parse_top_items(
        partner_data.get(int_col, ""),
        ["id", "name", "country", "type", "copubs", "share_ul", "share_int", "share_partner", "fwci"]
    )
    if int_items:
        int_df = pd.DataFrame(int_items)
        int_df["copubs"] = int_df["copubs"].apply(safe_int)
        int_df["share_ul"] = int_df["share_ul"].apply(safe_float)
        int_df["share_int"] = int_df["share_int"].apply(safe_float)
        int_df["share_partner"] = int_df["share_partner"].apply(safe_float)
        int_df["fwci"] = int_df["fwci"].apply(safe_float)
        
        int_display = int_df[["name", "country", "type", "copubs", "share_ul",  "share_partner","share_int", "fwci"]].copy()
        int_display.columns = [
            "Partner", "Country", "Type", "Co-pubs",
            f"% of UL's {level_label}",
            f"% of partner's {level_label}",
            "% of collab.",
            "Avg FWCI"
        ]
        int_display[f"% of UL's {level_label}"] = int_display[f"% of UL's {level_label}"] * 100
        int_display["% of collab."] = int_display["% of collab."] * 100
        int_display[f"% of partner's {level_label}"] = int_display[f"% of partner's {level_label}"] * 100
        int_display["Avg FWCI"] = int_display["Avg FWCI"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        
        st.dataframe(
            int_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                f"% of UL's {level_label}": st.column_config.ProgressColumn(
                    f"% of UL's {level_label}",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                ),
                "% of collab.": st.column_config.ProgressColumn(
                    "% of collab.",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                    help="Share of all UL co-publications with this partner",
                ),
                f"% of partner's {level_label}": st.column_config.ProgressColumn(
                    f"% of partner's {level_label}",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                ),
            }
        )
    else:
        st.info("No international partner data.")
    
    st.markdown("**Top 20 French Partners**")
    fr_col = [c for c in df_partners.columns if "top_fr_partners" in c][0]
    # Format: id:name:type:copubs:share_level:share_int:share_partner:fwci (8 fields, no country)
    fr_items = parse_top_items(
        partner_data.get(fr_col, ""),
        ["id", "name", "type", "copubs", "share_ul", "share_int", "share_partner", "fwci"]
    )
    if fr_items:
        fr_df = pd.DataFrame(fr_items)
        fr_df["copubs"] = fr_df["copubs"].apply(safe_int)
        fr_df["share_ul"] = fr_df["share_ul"].apply(safe_float)
        fr_df["share_int"] = fr_df["share_int"].apply(safe_float)
        fr_df["share_partner"] = fr_df["share_partner"].apply(safe_float)
        fr_df["fwci"] = fr_df["fwci"].apply(safe_float)
        
        fr_display = fr_df[["name", "type", "copubs", "share_ul", "share_partner", "share_int", "fwci"]].copy()
        fr_display.columns = [
            "Partner", "Type", "Co-pubs",
            f"% of UL's {level_label}",
            f"% of partner's {level_label}",
            "% of collab.",
            "Avg FWCI"
        ]
        fr_display[f"% of UL's {level_label}"] = fr_display[f"% of UL's {level_label}"] * 100
        fr_display["% of collab."] = fr_display["% of collab."] * 100
        fr_display[f"% of partner's {level_label}"] = fr_display[f"% of partner's {level_label}"] * 100
        fr_display["Avg FWCI"] = fr_display["Avg FWCI"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        
        st.dataframe(
            fr_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                f"% of UL's {level_label}": st.column_config.ProgressColumn(
                    f"% of UL's {level_label}",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                ),
                "% of collab.": st.column_config.ProgressColumn(
                    "% of collab.",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                    help="Share of all UL co-publications with this partner",
                ),
                f"% of partner's {level_label}": st.column_config.ProgressColumn(
                    f"% of partner's {level_label}",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                ),
            }
        )
    else:
        st.info("No French partner data.")

# =============================================================================
# Section 7: Strategic Reciprocity Chart (for OA taxonomy only)
# =============================================================================
if level in ["domain", "field", "subfield"] and partner_data is not None:
    st.markdown("---")
    st.markdown("### ⚖️ Strategic Reciprocity with Partners")
    
    st.markdown(f"""
    **How to read this chart**
    
    - Each bubble represents a partner institution. Its **size** is proportional to
      that partner's total publications in **{element_name}**.
    - The **vertical position** (y-axis) shows the share of UL's output in {element_name}
      that is co-authored with this partner.
    - The **horizontal position** (x-axis) shows the share of the **partner's**
      output in {element_name} that involves UL.
    - The grey **diagonal line** indicates balanced relationships.
    """)
    
    recip_col = [c for c in df_partners.columns if "reciprocity_partners" in c][0]
    # Format: id:name:country:type:copubs:share_ul:share_int:share_partner:partner_total:fwci (10 fields)
    recip_items = parse_top_items(
        partner_data.get(recip_col, ""),
        ["id", "name", "country", "type", "copubs", "share_ul", "share_int", "share_partner", "partner_total", "fwci"]
    )
    
    if recip_items:
        recip_df = pd.DataFrame(recip_items)
        recip_df["copubs"] = recip_df["copubs"].apply(safe_int)
        recip_df["share_ul"] = recip_df["share_ul"].apply(safe_float)
        recip_df["share_int"] = recip_df["share_int"].apply(safe_float)
        recip_df["share_partner"] = recip_df["share_partner"].apply(safe_float)
        recip_df["partner_total"] = recip_df["partner_total"].apply(safe_int)
        recip_df["fwci"] = recip_df["fwci"].apply(safe_float)
        
        # Filter out rows with no meaningful data
        recip_df = recip_df[(recip_df["share_ul"] > 0) | (recip_df["share_partner"] > 0)]
        recip_df = recip_df[recip_df["partner_total"] > 0]
        
        # Outlier toggle
        remove_outliers = st.checkbox(
            "Remove outliers (partner share > 100%)",
            value=False,
            help="Some partners may show >100% share due to data artifacts. Toggle to exclude them."
        )
        if remove_outliers:
            recip_df = recip_df[(recip_df["share_partner"] <= 1.0) & (recip_df["share_ul"] <= 1.0)]
        
        if not recip_df.empty:
            max_partners = min(50, len(recip_df))
            n_partners = st.slider(
                "Number of partners to display:",
                min_value=5,
                max_value=max_partners,
                value=min(30, max_partners),
            )
            
            recip_df = recip_df.nlargest(n_partners, "copubs")
            
            def geo_category(country):
                if country == "France":
                    return "France"
                if pd.isna(country) or country in ["", "None"]:
                    return "No country"
                return "International"
            
            recip_df["geo"] = recip_df["country"].apply(geo_category)
            
            fig_recip = px.scatter(
                recip_df,
                x="share_partner",
                y="share_ul",
                size="partner_total",
                size_max=40,
                color="geo",
                color_discrete_map={
                    "France": "blue",
                    "International": "red",
                    "No country": "#888888",
                },
                hover_name="name",
                custom_data=["country", "type", "copubs", "share_ul", "share_int", "share_partner", "partner_total", "fwci"],
            )
            
            fig_recip.update_traces(
                marker=dict(line=dict(color="black", width=0.5)),
                hovertemplate=(
                    "<b>%{hovertext}</b><br><br>"
                    "Country: %{customdata[0]}<br>"
                    "Type: %{customdata[1]}<br>"
                    "Co-publications: %{customdata[2]:,}<br>"
                    f"% of UL's {element_name}: " + "%{customdata[3]:.1%}<br>"
                    "% of collaboration: %{customdata[4]:.1%}<br>"
                    f"% of partner's {element_name}: " + "%{customdata[5]:.1%}<br>"
                    f"Partner's total in {element_name}: " + "%{customdata[6]:,}<br>"
                    "Avg FWCI: %{customdata[7]:.2f}<extra></extra>"
                )
            )
            
            max_val = max(recip_df["share_ul"].max(), recip_df["share_partner"].max()) * 1.1
            fig_recip.add_shape(
                type="line",
                x0=0, y0=0,
                x1=max_val, y1=max_val,
                line=dict(color="gray", dash="dash"),
            )
            
            fig_recip.update_layout(
                height=550,
                margin=dict(t=30, l=50, r=30, b=50),
                xaxis=dict(
                    title=f"Share of partner's {element_name} output",
                    tickformat=".0%",
                    range=[0, max_val],
                ),
                yaxis=dict(
                    title=f"Share of UL's {element_name} output",
                    tickformat=".0%",
                    range=[0, max_val],
                ),
                showlegend=False,
            )
            
            st.markdown(
                """
                <div style="margin-bottom: 0.5rem;">
                  <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background-color:blue;margin-right:4px;"></span>
                  <span style="margin-right:12px;">France</span>
                  <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background-color:red;margin-right:4px;"></span>
                  <span style="margin-right:12px;">International</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            st.plotly_chart(fig_recip, use_container_width=True)
        else:
            st.info("No reciprocity data available for this element.")
    else:
        st.info("No reciprocity data available.")

# =============================================================================
# Section 8: Top Authors
# =============================================================================
st.markdown("---")
st.markdown("### 👩‍🔬 Top 20 Authors")

author_data = get_author_data(level, element_id)

if author_data is not None:
    auth_col = [c for c in df_authors.columns if "top_authors" in c][0]
    auth_items = parse_top_items(
        author_data.get(auth_col, ""),
        ["id", "name", "orcid", "pubs", "pct", "fwci", "is_lorraine", "labs"]
    )
    
    if auth_items:
        auth_df = pd.DataFrame(auth_items)
        auth_df["pubs"] = auth_df["pubs"].apply(safe_int)
        auth_df["pct"] = auth_df["pct"].apply(safe_float)
        auth_df["fwci"] = auth_df["fwci"].apply(safe_float)
        auth_df["is_lorraine"] = auth_df["is_lorraine"].apply(lambda x: str(x).lower() == "true")
        auth_df["name"] = auth_df["name"].apply(add_spaces_to_name)
        
        fwci_col_name = f"Avg. FWCI in {level_label}"
        share_col_name = f"{level_label} share"
        
        auth_display = auth_df[["name", "orcid", "pubs", "pct", "fwci", "is_lorraine", "labs"]].copy()
        auth_display.columns = ["Author", "ORCID", "Pubs", share_col_name, fwci_col_name, "UL Affiliation", "Labs"]
        auth_display[share_col_name] = auth_display[share_col_name].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
        auth_display[fwci_col_name] = auth_display[fwci_col_name].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        auth_display["UL Affiliation"] = auth_display["UL Affiliation"].apply(lambda x: "✅" if x else "")
        auth_display["Labs"] = auth_display["Labs"].apply(lambda x: x.replace("/", " | ") if x else "")
        
        st.dataframe(auth_display, use_container_width=True, hide_index=True, height=500)
    else:
        st.info("No author data available.")
else:
    st.info("No author data available.")

# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.caption("Data: Université de Lorraine publications 2019-2023 | OpenAlex + custom topic modeling")