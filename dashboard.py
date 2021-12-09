import streamlit as st
from controller import EDA, train, infer
from model import multi_tab

#dashboard#
st.title("Card Fraud Detection Dashboard")
st.sidebar.title("Data Themes")

dashboard = multi_tab()

dashboard.add_tab("EDA", EDA)
dashboard.add_tab("training", train)
dashboard.add_tab("inference", infer)

dashboard.launch_view()
