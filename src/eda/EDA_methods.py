import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from matplotlib import colors
import matplotlib
from sklearn import metrics
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ..constants import DATASET_PATH
##############################################################################
@st.cache(allow_output_mutation=True)
def prepare_data():
    df = pd.read_csv(DATASET_PATH)
    Transaction_class = df["Class"].copy()
    Transaction_class[Transaction_class == 0] = "Clean transaction"
    Transaction_class[Transaction_class == 1] = "Fraudulent transaction"
    X = df.drop(labels= ["Class"],axis = 1)
    sc = StandardScaler()
    sc.fit(X)
    X_std = sc.transform(X)
    return df,  X_std, Transaction_class

df, X_std, Transaction_class = prepare_data()
##############################################################################


##############################################################################
def univariate_analysis(feature):
    data = df.loc[:,[feature]]
    mean = np.mean(data)[0]
    median = np.median(data)
    std = np.std(data)[0]
    max_v = data.max()[0]
    min_v = data.min()[0]
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    skewness = data.skew()
    kurtosis = data.kurtosis()
    analysis = {
    "Mean value" : mean,
    "Median value": median,
    "Standard deviation":std,
    "Max value" : max_v,
    "Min value": min_v,
    "First quartile" : Q1,
    "Third quartile" : Q3,
    "Interquartile": Q3 - Q1,
    "Skewness" : skewness,
    "Kurtosis" : kurtosis
                      }
    return pd.DataFrame.from_dict(analysis)

@st.cache(allow_output_mutation=True)
def save_univariate_analysis():
    path = os.path.join(os.getcwd(),"ressources")
    ressources = os.listdir(path)

    if "univariate_analysis.pickle" in ressources :
        with open(os.path.join(path,'univariate_analysis.pickle'), 'rb') as handle:
            dict_univariate_analysis = pickle.load(handle)
        return dict_univariate_analysis
    else :
        dict_univariate_analysis = {}
        for col in df.columns :
            dict_univariate_analysis[col] = univariate_analysis(col)
        with open(os.path.join(path,'univariate_analysis.pickle'), 'wb') as handle:
            pickle.dump(dict_univariate_analysis, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return dict_univariate_analysis
dict_univariate_analysis = save_univariate_analysis()
##############################################################################


##############################################################################
def plot_central_tendencies(feature):
    data =  df.loc[:,[feature]]
  
    mean = np.mean(data)[0]
    median = np.median(data)
    std = np.std(data)[0]
    interval = (-(mean + 3*std), (mean + 3*std))
    fig, ax = plt.subplots(1,2,figsize=(15, 8))
    palette =sns.set_palette(sns.color_palette(["green","red"]))
    sns.boxplot(data = df,x = "Class" ,y = feature ,linewidth=2,
                ax=ax[0], palette=palette, hue='Class'
               )
    data.plot(kind = "density" , xlim = interval,ax=ax[1],color="k",label = "feature density",linewidth=1.5)
                     
    ax[1].axvline(x=mean,
                color='red',label = "mean",linewidth=3)
                       
    ax[1].axvline(x=median,
                color='green',label = "median",linewidth=3)
    ax[1].legend( bbox_to_anchor=(1, 0.8),labels =("feature density","mean","median"))
    ax[1].set_title("Distribution of : {}".format(feature),fontsize = 16,y = 1.02)
    plt.tight_layout()
    return fig
@st.cache(allow_output_mutation=True)
def save_central_tendencies_plots():
    path = os.path.join(os.getcwd(),"ressources")
    ressources = os.listdir(path)
    if "plot_central_tendencies.pickle" in ressources :
        with open(os.path.join(path,"plot_central_tendencies.pickle"), 'rb') as handle:
            dict_plot_central_tendencies = pickle.load(handle)
        return dict_plot_central_tendencies
    else :
        dict_plot_central_tendencies = {}
        for col in df.columns :
            dict_plot_central_tendencies[col] = plot_central_tendencies(col)
        with open(os.path.join(path,'plot_central_tendencies.pickle'), 'wb') as handle:
            pickle.dump(dict_plot_central_tendencies, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return dict_plot_central_tendencies
dict_plot_central_tendencies = save_central_tendencies_plots()
##############################################################################


##############################################################################
@st.cache(allow_output_mutation=True)
def Class_distribution():
    fig, ax = plt.subplots(nrows= 1,figsize=(15, 8))
    graph = Transaction_class.value_counts().plot(kind = "bar",label ="clean" , ax = ax,rot=0 ,
                                          ylabel = "Transaction Count",
                                          color = ("green","red") 
                                         )
    ax.set_title("Transactions count per class",fontsize = 15,y = 1.02)

    p1 = graph.patches[0]
    height1 = p1.get_height()
    graph.text(p1.get_x()+p1.get_width()/2., height1 + 1,
            height1,ha="center")

    p2 = graph.patches[1]
    height2 = p2.get_height()
    graph.text(p2.get_x()+p2.get_width()/2., height2 + 1,
            height2 ,ha="center")
    return fig
##############################################################################


##############################################################################
@st.cache(allow_output_mutation=True)
def bivariate_plot(feature1,feature2):
        plt.figure(figsize=(15,8))
        s = sns.scatterplot(df[feature1], df[feature2],c=df["Class"],
                cmap=colors.ListedColormap(["green","red"])
                )
        plt.title("{} and {} relationship".format(feature1,feature2)) #title
        fig = s.get_figure()
        return fig
##############################################################################


##############################################################################
@st.cache(allow_output_mutation=True)
def correlation_heatmap():
    plt.figure(figsize=(32,16))
    s = sns.heatmap(df.corr(),cmap=plt.cm.Reds,annot=True,fmt='.1f')
    plt.title('Heatmap displaying the Pearson correlation coefficient',
            fontsize=13)
    fig = s.get_figure()
    return fig
##############################################################################


##############################################################################
@st.cache(allow_output_mutation=True)
def gram_matrix(gamma = 0.01):
    kmatrix = metrics.pairwise.rbf_kernel(df.drop(labels= "Class",axis = 1).iloc[:5000,:], gamma=gamma)
    kmatrix100 = kmatrix[:500, :500]
    fig, ax = plt.subplots(1,figsize=(15, 10))
    s = ax.pcolor(kmatrix100, cmap=matplotlib.cm.OrRd)
    fig.colorbar(s)
    fig.gca().invert_yaxis()
    fig.gca().xaxis.tick_top()
    return fig
##############################################################################


##############################################################################
@st.cache(allow_output_mutation=True)
def explained_variance_plot():
    pca = PCA()
    pca.fit_transform(X_std)
    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    fig, ax = plt.subplots(1,figsize=(15, 8))
    plt.bar(df.columns[:-1], exp_var_pca, alpha=0.5, align='center', label='Individual explained variance',color = "red")
    ax.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance',color = "red")
    ax.set_ylabel('Explained variance ratio')
    ax.set_xlabel('Principal component index')

    ax.legend(loc='best')
    fig.tight_layout()

    return fig