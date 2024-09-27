import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
import seaborn as sns
import matplotlib.pyplot as plt


def plot_silhouette_score(data, parameters=[2, 3, 4, 5, 10]):
    parameter_grid = ParameterGrid({"n_clusters": parameters})
    best_score = -1
    kmeans_model = KMeans()  # instantiating KMeans model
    silhouette_scores = []
    # evaluation based on silhouette_score
    for p in parameter_grid:
        kmeans_model.set_params(**p)  # set current hyper parameter
        kmeans_model.fit(
            data
        )  # fit model on wine dataset, this will find clusters based on parameter p
        ss = metrics.silhouette_score(
            data, kmeans_model.labels_
        )  # calculate silhouette_score
        silhouette_scores += [ss]  # store all the scores
        print("Parameter:", p, "Score", ss)
        # check p which has the best score
        if ss > best_score:
            best_score = ss
            best_grid = p
    # plotting silhouette score
    plt.bar(
        range(len(silhouette_scores)),
        list(silhouette_scores),
        align="center",
        color="#722f59",
        width=0.5,
    )
    plt.xticks(range(len(silhouette_scores)), list(parameters))
    plt.title("Silhouette Score", fontweight="bold")
    plt.xlabel("Number of Clusters")
    plt.show()


def plotar_matriz_corr(df):
    corr = df.corr()

    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap="YlGnBu")

    plt.show()


def plotar_scree_plot(pca_explained_variance_ratio_, figsize=(8, 5)):
    _ = plt.figure(figsize=figsize)

    plt.plot(np.cumsum(pca_explained_variance_ratio_), "ro-", linewidth=2)
    plt.title("Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalue")

    plt.show()


def plotar_matriz_corr(df):
    corr = df.corr()

    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap="YlGnBu")

    plt.show()
