import streamlit as st
import streamlit.components.v1 as components


# Set page title
st.set_page_config(
    page_title="Vitality Lung Predictor....",
    initial_sidebar_state="expanded",
)
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
img{border-radius: 10px;}
</style> """, unsafe_allow_html=True)
st.write('<style>div.row-widget.stMarkdown { font-size: 24px; }</style>', unsafe_allow_html=True)



st.markdown("## Vitality Lung Predictor ")
st.write("Vitality Lung Predictor, a cutting-edge app harnessing advanced AI for precise lung cancer prediction and classification. Swiftly analyzing medical images, it distinguishes between benign and malignant tumors, further categorizing into specific types such as large cell carcinoma, adenocarcinoma, squamous, and normal lung tissue. Empowering early detection and personalized treatment strategies.")


