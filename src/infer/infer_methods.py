import streamlit as st
from src.constants import INFERENCE_EXAMPLE
import time
import requests
import pandas as pd


def init_db(conn):
    sql_create_projects_table = """ CREATE TABLE IF NOT EXISTS inference_results (
                                    id integer PRIMARY KEY,
                                    feature_11 integer,
                                    feature_13 integer,
                                    feature_15 integer,
                                    amount integer,
                                    Output integer
                                ); """
    c = conn.cursor()
    c.execute(sql_create_projects_table)
    conn.commit()


def build_slidebar(conn):
    st.sidebar.header("Features Configuration")
    feature_11 = st.slider('Transaction Feature 11', -
                           10.0, 10.0, step=0.001, value=-4.075)
    feature_13 = st.slider('Transaction Feature 13', -
                           10.0, 10.0, step=0.001, value=0.963)
    feature_15 = st.slider('Transaction Feature 15', -
                           10.0, 10.0, step=0.001, value=2.630)
    amount = st.number_input(
        'Transaction Amount', value=1000, min_value=0, max_value=int(1e10), step=100)

    INFERENCE_EXAMPLE[11] = feature_11
    INFERENCE_EXAMPLE[13] = feature_13
    INFERENCE_EXAMPLE[15] = feature_15
    INFERENCE_EXAMPLE[28] = amount

    infer = st.button('Run Fraud Inference')
    if infer:
        with st.spinner('Running inference...'):
            time.sleep(1)
            try:
                result = requests.post(
                    'http://localhost:5000/api/inference',
                    json=INFERENCE_EXAMPLE
                )

                if int(int(result.text) == 1):
                    st.success('Done!')
                    st.metric(label="Status", value="Transaction: Fraudulent")

                else:
                    st.success('Done!')
                    st.metric(label="Status", value="Transaction: Clear")
            except Exception as e:
                st.error('Failed to call Inference API!')
                st.exception(e)

    try:
        sql = ''' INSERT INTO inference_results(feature_11,feature_13,feature_15,amount,Output)
                    VALUES(?,?,?,?,?) '''
        c = conn.cursor()
        c.execute(sql, (feature_11, feature_13,
                        feature_15, amount, int(result.text)))
        st.success('Results saved to database')
        conn.commit()
    except:
        st.error('run an Inference')


def get_data(conn):
    df = pd.read_sql("SELECT * FROM inference_results", con=conn)
    return df


def display_data(conn):
    if st.checkbox("Display data in sqlite databse"):
        st.table(get_data(conn))
