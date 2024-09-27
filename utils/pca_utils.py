from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Vamos considerar 2 componentes principais
def calcular_pca(X, n_components=5, prefix="PCA_"):
    pca = PCA(n_components=n_components)
    pca.fit(X)

    pca_transformed = pca.transform(X)
    df_pca = pd.DataFrame(pca_transformed)

    return pca, df_pca


def calcular_cargas_fatoriais(pca, df_cats, prefix="PCA_"):
    for i, componente in enumerate(pca.components_):
        df_cats[f"{prefix}{i}"] = componente

    return df_cats.sort_values(f"{prefix}0", ascending=False)
