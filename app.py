from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="Lorraine Explorer v1", layout="wide")
st.title("Lorraine Explorer v1")

st.write("Use the sidebar to open a view:")
st.page_link("pages/1_🏭_Lab_Overview.py", label="🏭 Lab_Overview")