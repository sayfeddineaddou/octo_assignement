import streamlit as st
import time
from PIL import Image
import sqlite3
from model import eda_header, eda_info, training_header, training_info, inference_info, inference_header
from src.eda.EDA_methods import df, dict_univariate_analysis, dict_plot_central_tendencies,\
    Class_distribution, bivariate_plot,\
    correlation_heatmap, gram_matrix, explained_variance_plot
from src.infer.infer_methods import build_slidebar, display_data, init_db
from src.training.train_pipeline import TrainingPipeline
from src.constants import INFERENCE_EXAMPLE, CM_PLOT_PATH

########## Headers Wrapper ##########


def decorator(header, info):
    def header_info(tab_option):

        def wrapper(**kwargs):

            st.header(header)
            st.info(info)
            tab_option()

        return wrapper
    return header_info
#####################################


################ EDA ################
@decorator(eda_header, eda_info)
def EDA():

    st.subheader("Univariate Analysis")
    option1 = st.selectbox("Choose variable", options=df.columns)
    st.table(data=dict_univariate_analysis[option1].iloc[:, [0, 1, 2, 3, 4]])
    st.table(data=dict_univariate_analysis[option1].iloc[:, [5, 6, 7, 8, 9]])

    st.subheader("Central Tendencies")
    # show_central_tendencies = st.button('Show',key = 2)
    # if show_central_tendencies :
    st.pyplot(dict_plot_central_tendencies[option1], clear_figure=False)

    st.subheader("Class distribution")
    # show_class_distribution = st.button('Show',key = 3)
    # if show_class_distribution :
    st.pyplot(Class_distribution(), clear_figure=False)

    st.subheader("Bivariate Analysis")
    option2 = st.multiselect("Choose two variables", options=df.columns)
    show_bivariate_analysis = st.button('Show', key=4)
    if show_bivariate_analysis:
        try:
            st.pyplot(bivariate_plot(
                option2[0], option2[1]), clear_figure=False)
        except:
            st.error('Please Choose two variables !')
    st.subheader("Correlation HeatMap")
    st.pyplot(correlation_heatmap(), clear_figure=False)

    st.subheader("Gram Matrix")
    bandwidth = st.number_input("Choose RBF kernel bandwidth", value=100)
    st.write('The current bandwidth is ', bandwidth)
    st.pyplot(gram_matrix(gamma=1/bandwidth), clear_figure=False)

    st.subheader("Explained Variance")
    st.pyplot(explained_variance_plot(), clear_figure=False)


########## Model Training ###########
@decorator(training_header, training_info)
def train():
    name = st.text_input('model name', placeholder='decisiontree')
    serialize = st.checkbox('Save model')
    train = st.button('Train Model')

    if train:
        with st.spinner('training model, please wait...'):
            time.sleep(1)
            try:
                tp = TrainingPipeline()
                tp.train(serialize=serialize, model_name=name)
                tp.render_confusion_matrix(plot_name=name)
                accuracy, precision, recall, f1 = tp.get_model_perfomance()
                col1, col2, col3, col4 = st.columns(4)

                col1.metric(label="Accuracy score",
                            value=str(round(accuracy, 4)))
                col2.metric(label="Precision score",
                            value=str(round(precision, 4)))
                col3.metric(label="Recall  score",
                            value=str(round(recall, 4)))
                col4.metric(label="F1 score",
                            value=str(round(f1, 4)))
                st.image(Image.open(CM_PLOT_PATH))

            except Exception as e:
                st.error('Failed to train model!')
                st.exception(e)
#####################################


############# Inference #############
@decorator(inference_header, inference_info)
def infer():
    conn = sqlite3.connect("inference.db")
    init_db(conn=conn)
    st.success('table created!')
    build_slidebar(conn=conn)
    display_data(conn=conn)
    