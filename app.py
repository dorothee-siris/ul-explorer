from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="Lorraine Explorer v1", layout="wide")
st.title("Lorraine Explorer v1")

st.write("Use the sidebar to open a view:")
st.page_link("pages/1_🏭_Lab_Overview.py", label="🏭 Lab_Overview")
st.page_link("pages/1b_🏭_Lab_Collaboration.py", label="🏭 Lab Collaboration")
st.page_link("pages/2_🔬_Topic_View.py", label="🔬 Topic View")
st.page_link("pages/3_🤝_Partners_Overview.py", label="🤝 Partners Overview")
st.page_link("pages/3_🤝_Partners_Drill_Down.py", label="🤝 Partners Drill Down")