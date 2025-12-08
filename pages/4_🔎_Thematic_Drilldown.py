"""
Thematic Drill-Down - Detailed exploration of a specific domain, field, subfield, or research topic.

Sections:
1. Selector (level + element)
2. Topline KPIs
3. Sublevel breakdown table (for OA taxonomy)
4. Time evolution charts (absolute + stacked)
5. Contribution analysis (research topics, departments, labs)
6. Partner tables (international, national)
7. Strategic reciprocity chart (for OA taxonomy)
8. Top authors table
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
    DOMAIN_NAMES_ORDERED,
    get_domain_id_to_name,
    get_field_id_to_name,
    get_field_id_to_domain_id,
    get_subfield_id_to_name,
    get_subfield_id_to_domain_id,
    get_domain_color,
    render_domain_legend,
    safe_float,
    safe_int,
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
    "facility": "#f28e2b",
    "other": "#76b7b2",
}

# =============================================================================
# Load data
# =============================================================================
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
        return df.set_index("structure_key")[["Structure name", "Structure type"]].to_dict("index")
    except:
        return {}

df_overview = load_thematic_overview()
df_sublevels = load_thematic_sublevels()
df_contributions = load_thematic_contributions()
df_partners = load_thematic_partners()
df_authors = load_thematic_authors()
lab_info = load_lab_info()

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
    """Parse pipe-separated items with colon-separated fields."""
    if pd.isna(blob) or not str(blob).strip():
        return []
    results = []
    for item in str(blob).split("|"):
        parts = item.split(":")
        if len(parts) >= len(expected_fields):
            row = {field: parts[i] for i, field in enumerate(expected_fields)}
            results.append(row)
    return results

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
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:16px;">'
        f'<span style="width:14px;height:14px;background:{color};border-radius:3px;margin-right:6px;"></span>'
        f'{stype.title()}</span>'
        for stype, color in STRUCTURE_TYPE_COLORS.items()
    )
    st.markdown(f'<div style="margin:8px 0 16px 0;">{items}</div>', unsafe_allow_html=True)

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
if level != "research_topic":
    render_domain_legend()

# =============================================================================
# Section 2: Topline KPIs
# =============================================================================
st.markdown("---")

# Volume & Growth section
st.markdown("#### 📊 Volume & Growth")
kpi_cols1 = st.columns(4)

with kpi_cols1[0]:
    st.metric("Publications", f"{int(element_data['pubs_total']):,}")

with kpi_cols1[1]:
    st.metric("% of UL Total", format_pct(element_data['pubs_pct_of_ul']))

with kpi_cols1[2]:
    st.metric("CAGR 2019-23", format_cagr(element_data['cagr_2019_2023']))

with kpi_cols1[3]:
    st.metric("% SDG-related", format_pct(element_data['pct_sdg']))

# Impact section
st.markdown("#### 🎯 Impact & Excellence")
kpi_cols2 = st.columns(4)

with kpi_cols2[0]:
    fwci_median = element_data['fwci_median']
    fwci_median_str = f"{fwci_median:.2f}" if pd.notna(fwci_median) else "—"
    st.metric("Median FWCI", fwci_median_str)

with kpi_cols2[1]:
    fwci_mean = element_data['fwci_mean']
    fwci_mean_str = f"{fwci_mean:.2f}" if pd.notna(fwci_mean) else "—"
    st.metric("Avg. FWCI", fwci_mean_str)

with kpi_cols2[2]:
    st.metric("% Top 10%", format_pct(element_data['pct_top10']))

with kpi_cols2[3]:
    st.metric("% Top 1%", format_pct(element_data['pct_top1']))

# Collaboration & Funding section
st.markdown("#### 🤝 Collaboration & Funding")
kpi_cols3 = st.columns(4)

with kpi_cols3[0]:
    st.metric("🌍 % International", format_pct(element_data['pct_international']))

with kpi_cols3[1]:
    st.metric("🏢 % Company", format_pct(element_data['pct_company']))

with kpi_cols3[2]:
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
        
        # Add domain info for coloring
        if level == "domain":
            df_sub["domain_name"] = element_name
        elif level == "field":
            parent_domain_id = field_id2domain.get(int(element_id), 0)
            df_sub["domain_name"] = domain_id2name.get(parent_domain_id, "Other")
        else:
            parent_domain_id = subfield_id2domain.get(int(element_id), 0)
            df_sub["domain_name"] = domain_id2name.get(parent_domain_id, "Other")
        
        sub_table = []
        for _, row in df_sub.iterrows():
            sub_table.append({
                "Name": row["child_name"],
                "Pubs": int(row["pubs_total"]),
                f"% of {level_label}": row["pubs_pct_of_parent"],
                "% ISITE": row["pct_isite"],
                "% Top 10%": format_pct(row["pct_top10"]),
                "% Top 1%": format_pct(row["pct_top1"]),
                "% Int'l": format_pct(row["pct_international"]),
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
                f"% of {level_label}": st.column_config.ProgressColumn(
                    f"% of {level_label}",
                    min_value=0,
                    max_value=1,
                    format="%.1f%%",
                ),
                "% ISITE": st.column_config.ProgressColumn(
                    "% ISITE",
                    min_value=0,
                    max_value=1,
                    format="%.1f%%",
                ),
            }
        )
        
        # =============================================================================
        # Section 4: Time Evolution Charts
        # =============================================================================
        st.markdown(f"### 📈 Time Evolution of {child_level_label}s")
        
        # Parse year counts for each sublevel
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
            # Get top 10 sublevels by total volume
            top_names = df_sub.nlargest(10, "pubs_total")["child_name"].tolist()
            df_time_top = df_time[df_time["Name"].isin(top_names)]
            
            # Add "Other" for remaining
            df_time_other = df_time[~df_time["Name"].isin(top_names)].groupby("Year")["Count"].sum().reset_index()
            df_time_other["Name"] = "Other"
            
            df_time_plot = pd.concat([df_time_top, df_time_other], ignore_index=True)
            
            # Shared color mapping
            all_names = top_names + ["Other"]
            color_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set2
            color_map = {name: color_palette[i % len(color_palette)] for i, name in enumerate(all_names)}
            
            # Legend (shared)
            st.markdown("**Legend:**")
            legend_items = " · ".join([
                f'<span style="color:{color_map[name]}">●</span> {name}'
                for name in all_names if name in df_time_plot["Name"].unique()
            ])
            st.markdown(f'<div style="margin-bottom:16px;font-size:0.9em;">{legend_items}</div>', unsafe_allow_html=True)
            
            # Absolute values chart
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
                showlegend=False,
                xaxis=dict(dtick=1),
                yaxis_title="Publications",
            )
            st.plotly_chart(fig_abs, use_container_width=True)
            
            # Stacked area chart
            st.markdown("**Relative share (100% stacked)**")
            df_time_pct = df_time_plot.copy()
            year_totals = df_time_pct.groupby("Year")["Count"].transform("sum")
            df_time_pct["Percentage"] = (df_time_pct["Count"] / year_totals * 100).fillna(0)
            
            fig_stack = px.area(
                df_time_pct,
                x="Year",
                y="Percentage",
                color="Name",
                color_discrete_map=color_map,
                groupnorm="percent",
            )
            fig_stack.update_layout(
                height=400,
                margin=dict(t=30, l=50, r=30, b=50),
                showlegend=False,
                xaxis=dict(dtick=1),
                yaxis=dict(title="Share (%)", range=[0, 100]),
            )
            st.plotly_chart(fig_stack, use_container_width=True)
    else:
        st.info("No sublevel data available.")

# =============================================================================
# Section 5: Contribution Analysis
# =============================================================================
st.markdown("---")
st.markdown("### 🏗️ Contribution Analysis")

contrib_data = get_contribution_data(level, element_id)

if contrib_data is not None:
    # Top Research Topics (skip for research_topic level)
    if level != "research_topic":
        st.markdown("**Top 5 Research Topics**")
        rt_items = parse_top_items(
            contrib_data.get("top_research_topics", ""),
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
                height=300,
                margin=dict(t=10, l=10, r=10, b=10),
                xaxis_title="Publications",
                yaxis_title="",
            )
            st.plotly_chart(fig_rt, use_container_width=True)
        else:
            st.info("No research topic data.")
    
    # Department Distribution
    st.markdown("**Department Distribution**")
    dept_items = parse_top_items(
        contrib_data.get("department_breakdown", ""),
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
    
    # Top 10 Labs
    st.markdown("**Top 10 Labs / Structures**")
    render_structure_type_legend()
    
    lab_items = parse_top_items(
        contrib_data.get("top_labs", ""),
        ["ror", "count", "pct"]
    )
    if lab_items:
        lab_df = pd.DataFrame(lab_items)
        lab_df["count"] = lab_df["count"].apply(safe_int)
        lab_df["pct"] = lab_df["pct"].apply(safe_float)
        
        # Join with lab info
        lab_df["name"] = lab_df["ror"].apply(
            lambda x: lab_info.get(x, {}).get("Structure name", x) if x in lab_info else x
        )
        lab_df["type"] = lab_df["ror"].apply(
            lambda x: lab_info.get(x, {}).get("Structure type", "other") if x in lab_info else "other"
        )
        lab_df["color"] = lab_df["type"].apply(lambda x: STRUCTURE_TYPE_COLORS.get(x, STRUCTURE_TYPE_COLORS["other"]))
        
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
    # International Partners
    st.markdown("**Top 10 International Partners**")
    int_col = [c for c in df_partners.columns if "top_int_partners" in c][0]
    int_items = parse_top_items(
        partner_data.get(int_col, ""),
        ["id", "name", "country", "type", "copubs", "pct", "fwci"]
    )
    if int_items:
        int_df = pd.DataFrame(int_items)
        int_df["copubs"] = int_df["copubs"].apply(safe_int)
        int_df["pct"] = int_df["pct"].apply(safe_float)
        int_df["fwci"] = int_df["fwci"].apply(safe_float)
        
        int_display = int_df[["name", "country", "type", "copubs", "pct", "fwci"]].copy()
        int_display.columns = ["Partner", "Country", "Type", "Co-pubs", f"% of {element_name}", "Avg FWCI"]
        int_display[f"% of {element_name}"] = int_display[f"% of {element_name}"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
        int_display["Avg FWCI"] = int_display["Avg FWCI"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        
        st.dataframe(int_display, use_container_width=True, hide_index=True)
    else:
        st.info("No international partner data.")
    
    # French Partners
    st.markdown("**Top 10 French Partners**")
    fr_col = [c for c in df_partners.columns if "top_fr_partners" in c][0]
    fr_items = parse_top_items(
        partner_data.get(fr_col, ""),
        ["id", "name", "country", "type", "copubs", "pct", "fwci"]
    )
    if fr_items:
        fr_df = pd.DataFrame(fr_items)
        fr_df["copubs"] = fr_df["copubs"].apply(safe_int)
        fr_df["pct"] = fr_df["pct"].apply(safe_float)
        fr_df["fwci"] = fr_df["fwci"].apply(safe_float)
        
        fr_display = fr_df[["name", "type", "copubs", "pct", "fwci"]].copy()
        fr_display.columns = ["Partner", "Type", "Co-pubs", f"% of {element_name}", "Avg FWCI"]
        fr_display[f"% of {element_name}"] = fr_display[f"% of {element_name}"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
        fr_display["Avg FWCI"] = fr_display["Avg FWCI"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        
        st.dataframe(fr_display, use_container_width=True, hide_index=True)
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
        - Bubbles **to the right** of the diagonal: UL is more important to the partner than vice versa.
        - Bubbles **to the left** of the diagonal: The partner is more important to UL than vice versa.
    """)
    
    recip_col = [c for c in df_partners.columns if "reciprocity_partners" in c][0]
    recip_items = parse_top_items(
        partner_data.get(recip_col, ""),
        ["id", "name", "country", "type", "copubs", "share_ul", "share_partner", "partner_total"]
    )
    
    if recip_items:
        recip_df = pd.DataFrame(recip_items)
        recip_df["copubs"] = recip_df["copubs"].apply(safe_int)
        recip_df["share_ul"] = recip_df["share_ul"].apply(safe_float)
        recip_df["share_partner"] = recip_df["share_partner"].apply(safe_float)
        recip_df["partner_total"] = recip_df["partner_total"].apply(safe_int)
        
        # Filter out rows with zero shares
        recip_df = recip_df[(recip_df["share_ul"] > 0) | (recip_df["share_partner"] > 0)]
        
        if not recip_df.empty:
            # Slider for number of partners
            max_partners = min(50, len(recip_df))
            n_partners = st.slider(
                "Number of partners to display:",
                min_value=5,
                max_value=max_partners,
                value=min(30, max_partners),
            )
            
            recip_df = recip_df.nlargest(n_partners, "copubs")
            
            # Geo category for coloring
            def geo_category(country):
                if country == "France":
                    return "France"
                if pd.isna(country) or country in ["", "None"]:
                    return "No country"
                return "International"
            
            recip_df["geo"] = recip_df["country"].apply(geo_category)
            
            # Create scatter plot
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
                custom_data=["country", "type", "copubs", "share_ul", "share_partner", "partner_total"],
            )
            
            fig_recip.update_traces(
                marker=dict(line=dict(color="black", width=0.5)),
                hovertemplate=(
                    "<b>%{hovertext}</b><br><br>"
                    "Country: %{customdata[0]}<br>"
                    "Type: %{customdata[1]}<br>"
                    "Co-publications: %{customdata[2]:,}<br>"
                    f"Share of UL's {element_name}: " + "%{customdata[3]:.1%}<br>"
                    f"Share of partner's {element_name}: " + "%{customdata[4]:.1%}<br>"
                    f"Partner's total in {element_name}: " + "%{customdata[5]:,}<extra></extra>"
                )
            )
            
            # Add diagonal line
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
            
            # Custom legend
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
        auth_df["is_lorraine"] = auth_df["is_lorraine"].apply(lambda x: x.lower() == "true" if isinstance(x, str) else bool(x))
        
        auth_display = auth_df[["name", "orcid", "pubs", "pct", "fwci", "is_lorraine", "labs"]].copy()
        auth_display.columns = ["Author", "ORCID", "Pubs", f"% of {element_name}", "Avg FWCI", "UL Affiliation", "Labs"]
        auth_display[f"% of {element_name}"] = auth_display[f"% of {element_name}"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
        auth_display["Avg FWCI"] = auth_display["Avg FWCI"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
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