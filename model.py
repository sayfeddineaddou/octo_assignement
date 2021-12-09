import streamlit as st
eda_header = "Exploratory Data Analysis"
eda_info = "In this section, you are invited to create insightful graphs \
    about the card fraud dataset that you were provided."
# ------------------------------------------------------------------------
training_header = "Model Training"
training_info = "Before you proceed to training your model. Make sure you \
have checked your training pipeline code and that it is set properly."
# ------------------------------------------------------------------------
inference_info = "Fraud Inference"
inference_header = "This section simplifies the inference process. \
You can tweak the values of feature 1, 2, 19, \
and the transaction amount and observe how your model reacts to these changes."
# ------------------------------------------------------------------------


class multi_tab:
    """
    class model that will create and handle the tabs used on the dashboard view
    """

    def __init__(self):
        """generate a list that will store pages"""
        self.tabs = []

    def add_tab(self, option, function):
        """
        methode that add new tabs to our dashboard
        """
        self.tabs.append(
            {
                "option": option,
                "function": function
            }
        )

    def launch_view(self):
        tab = st.sidebar.selectbox(
            "Options",
            self.tabs,
            format_func=lambda tab: tab["option"]
        )
        print(tab)
        tab["function"]()
