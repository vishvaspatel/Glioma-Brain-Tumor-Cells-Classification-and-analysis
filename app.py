import streamlit as st

# Set page configuration
st.set_page_config(page_title="Image Analysis App", layout="wide")

# Define pages as a list
pages = [
    st.Page("pages/segmentation_classification.py", title="Segmentation + Classification"),
    st.Page("pages/cell_image_placement.py", title="Mix Image Generator"),
]

# Render navigation
st.navigation(pages).run()