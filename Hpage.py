import streamlit as st
import streamlit.components.v1 as components

# Set page title
st.set_page_config(
    page_title="Lung Cancer Detection",
    initial_sidebar_state="expanded",
)
st.write('<style>div.row-widget.stMarkdown { font-size: 24px; }</style>', unsafe_allow_html=True)

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
components.html(
    """
    <style>
        #effect{
            margin:0px;
            padding:0px;
            font-family: "Source Sans Pro", sans-serif;
            font-size: max(8vw, 20px);
            font-weight: 700;
            top: 0px;
            right: 25%;
            position: fixed;
            background: -webkit-linear-gradient(0.25turn,blue,Lime);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        p{
            font-size: 2rem;
        }
    </style>
    <p id="effect">Lung Cancer Detection</p>
    """,
    height=69,
)



def page_layout():
   
    st.write("Lung Cancer Detection is an app developed by SVEC to detect the lung cancer")
    st.write("Developed By SVEC Students")
    st.markdown("## Benefits:")
    st.write("- Increased Survival Rates and Improved Quality of Life")
    st.write("- Accessible from anywhere, anytime")
    
    st.markdown("## Use:")
    st.write("- Improved Efficiency")
    st.write("- Lung Cancer Detection System is utilized for the early identification and diagnosis of lung cancer.The system is designed to identify potential signs of lung cancer at an early stage when the disease is more treatable and the chances of successful intervention and recovery are higher ")
  
# Render page layout
page_layout()
