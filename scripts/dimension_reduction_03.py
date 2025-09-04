import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import umap
import umap.plot
from IPython.display import display
from matplotlib import offsetbox
from matplotlib.image import imread
from sklearn.datasets import make_swiss_roll
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding

# Configuration pour un affichage plus riche
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

pio.renderers.default = 'browser'

def display_dataframe_info(df, title="DataFrame Info"):
    """Affiche les informations du DataFrame de manière formatée"""
    print("=" * 80)
    print(f"📊 {title}")
    print("=" * 80)
    
    # Informations de base
    print(f"📋 Forme du DataFrame: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"💾 Utilisation mémoire: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print()
    
    # Types de données
    print("🔍 Types de données:")
    print("-" * 40)
    for col, dtype in df.dtypes.items():
        print(f"  {col:<25} | {dtype}")
    print()
    
    # Valeurs manquantes
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("❌ Valeurs manquantes:")
        print("-" * 40)
        for col, missing in missing_values.items():
            if missing > 0:
                percentage = (missing / len(df)) * 100
                print(f"  {col:<25} | {missing:>6} ({percentage:>5.1f}%)")
        print()
    else:
        print("✅ Aucune valeur manquante détectée")
        print()
    
    # Aperçu des données
    print("👀 Aperçu des données (5 premières lignes):")
    print("-" * 40)
    print(df.head().to_string())
    print()
    
    # Statistiques descriptives pour les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print("📈 Statistiques descriptives (colonnes numériques):")
        print("-" * 40)
        print(df[numeric_cols].describe().to_string())
        print()
    
    # Valeurs uniques pour les colonnes catégorielles (limitées)
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print("🏷️  Aperçu des valeurs uniques (colonnes catégorielles):")
        print("-" * 40)
        for col in categorical_cols[:5]:  # Limite à 5 colonnes pour éviter l'encombrement
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count} valeurs uniques")
            if unique_count <= 10:
                print(f"    Valeurs: {list(df[col].unique())}")
            else:
                print(f"    Exemples: {list(df[col].unique()[:5])}...")
        print()


# Fonction pour afficher des informations spécifiques sur les colonnes
def explore_column(df, column_name):
    """Explore une colonne spécifique du DataFrame"""
    if column_name not in df.columns:
        print(f"❌ La colonne '{column_name}' n'existe pas.")
        return
    
    print(f"\n🔍 Exploration de la colonne: {column_name}")
    print("-" * 50)
    print(f"Type: {df[column_name].dtype}")
    print(f"Valeurs uniques: {df[column_name].nunique()}")
    print(f"Valeurs manquantes: {df[column_name].isnull().sum()}")
    
    if df[column_name].dtype == 'object':
        print("\nValeurs les plus fréquentes:")
        print(df[column_name].value_counts().head(10))
    elif df[column_name].dtype in ['int64', 'float64']:
        print(f"\nMin: {df[column_name].min()}")
        print(f"Max: {df[column_name].max()}")
        print(f"Moyenne: {df[column_name].mean():.2f}")
        print(f"Médiane: {df[column_name].median():.2f}")


# chargement des données
print("🔄 Chargement des données...")
try:
    df_train = pd.read_csv('data/fashion-mnist_train.csv')
    df_test = pd.read_csv('data/fashion-mnist_test.csv')
    display(df_train.info())
    display(df_test.info())
    print("✅ Données chargées avec succès!")
    print()
    df_train = df_train[0:6000]
    df_test = df_test[0:1000]
    display(df_train.info())
    display(df_test.info())
    print("✅ Données réduites pour accélérer les calculs!")


    # # On affiche une feuille plane en 3D
    # #Définir les dimensions de la feuille
    # longueur = 80
    # largeur = 24

    # # Générer des coordonnées x et y aléatoires pour les nuages de points
    # n_points = 5000
    # np.random.seed(42)  # Pour reproductibilité
    # x = np.random.rand(n_points) * longueur
    # y = np.random.rand(n_points) * largeur

    # # Coordonnées z pour obtenir une feuille plane
    # z = np.zeros_like(x)  

    # # Créer le graphique 3D
    # fig = go.Figure()

    # # Utiliser la coordonnée x comme valeur de couleur pour le dégradé arc-en-ciel
    # fig.add_trace(go.Scatter3d(x=x - 40, y=y, z=z, mode='markers', marker=dict(size=3, color=x, colorscale='Spectral', opacity=0.8)))

    # # Configurer les étiquettes des axes
    # fig.update_layout(scene=dict(
    #     xaxis_title='X',
    #     yaxis_title='Y',
    #     zaxis_title='Z'
    # ), scene_aspectmode='manual', scene_aspectratio=dict(x=2, y=0.5, z=1))

    # # Titre du graphique
    # fig.update_layout(title='Feuille plane vue en 3D', width=800, height=600)
    
    # # Afficher le graphique
    # fig.show()

    # # On affiche une feuille enroulée sur elle-même
    # # Générer les données de la feuille enroulée (Swiss Roll)
    # n_samples = 5000
    # X, color = make_swiss_roll(n_samples)

    # fig = go.Figure()

    # fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers', marker=dict(size=3, color=color, colorscale='Spectral')))

    # fig.update_layout(scene=dict(
    #     xaxis_title='X',
    #     yaxis_title='Y',
    #     zaxis_title='Z'
    # ), scene_aspectmode='manual', scene_aspectratio=dict(x=2, y=0.5, z=1))

    # fig.update_layout(title='Feuille enroulée sur elle-même (transfomation non linéaire)', width=800, height=600)

    # fig.show()

    # # On projette la feuille enroulée sur les 2 axes principaux de la PCA
    # pca = PCA(n_components = 2)

    # X_2D = pca.fit_transform(X)

    # fig = plt.figure()

    # ax = fig.add_subplot(111)
    # ax.scatter(X_2D[:, 0], X_2D[:, 1], cmap=plt.cm.Spectral)

    # ax.set_xlabel('PCA 1')
    # ax.set_ylabel('PCA 2')

    # ax.set_title("Feuille enroulée projetée sur les 2 axes de la PCA")
    # plt.show()

    # # On affiche une feuille enroulée sur elle-même avec le plan de projection
    # # Générer les données de la feuille enroulée (Swiss Roll)
    # n_samples = 5000
    # X, color = make_swiss_roll(n_samples)

    # fig = go.Figure()

    # fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers', marker=dict(size=3, color=color, colorscale='Spectral')))

    # y_plane = 30
    # x_plane = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    # z_plane = np.linspace(min(X[:, 2]), max(X[:, 2]), 100)
    # x_plane, z_plane = np.meshgrid(x_plane, z_plane)
    # y_plane = np.ones_like(x_plane) * y_plane

    # fig.add_trace(go.Surface(x=x_plane, y=y_plane, z=z_plane, opacity=0.5))

    # fig.update_layout(scene=dict(
    #     xaxis_title='X',
    #     yaxis_title='Y',
    #     zaxis_title='Z'
    # ), scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=1, z=1))

    # fig.update_layout(title='Feuille enroulée sur elle-même avec le plan de projection en violet et jaune', width=800, height=600)

    # fig.show()

    # # On projette la feuille enroulée sur les 2 axes principaux de l'Isomap
    # # Générer les données de la feuille enroulée (Swiss Roll)
    # n_samples = 5000
    # X, color = make_swiss_roll(n_samples)

    # isomap = Isomap(n_neighbors=12, n_components=2)
    # X_isomap = isomap.fit_transform(X)

    # fig = go.Figure()

    # fig.add_trace(go.Scatter3d(x=X_isomap[:, 0], y=X_isomap[:, 1], z=np.zeros(X.shape[0]), mode='markers', marker=dict(size=5, color=X_isomap[:, 0], colorscale='Spectral')))

    # fig.update_layout(scene=dict(
    #     xaxis_title='X',
    #     yaxis_title='Y',
    #     zaxis_title='Z',
    #     aspectmode='manual',
    #     aspectratio=dict(x=2, y=0.5, z=1) 
    # ))

    # fig.update_layout(title="Feuille déroulée par la méthode Isomap", width=800, height=600)

    # fig.show()

    # Séparation des features et de la target
    df_bis = pd.concat([df_train, df_test], axis=0)
    target = df_bis['label']
    data = df_bis.drop('label',axis=1)

    # On projette les images sur les 2 axes principaux de la LLE
    # lle = LocallyLinearEmbedding(n_neighbors=50, n_components=2, method='modified', random_state = 42)
    # dataLLE = lle.fit_transform(data)

    # fig = plt.figure()

    # ax = fig.add_subplot(111)
    # ax.scatter(dataLLE[:, 0], dataLLE[:, 1],  c = target, cmap=plt.cm.Spectral, alpha = .7, s = 4)

    # #ax.set_xlabel('LL 1')
    # #ax.set_ylabel('LL 2')

    # ax.set_title("Manifold 2D identifié par la LLE")
    # plt.show()

    def plot_components(data, model, images=None, ax=None,
                        thumb_frac=0.05, cmap='gray_r', prefit = False):
        ax = ax or plt.gca()
        
        if not prefit :
            proj = model.fit_transform(data)
        else:
            proj = data
        ax.plot(proj[:, 0], proj[:, 1], '.b')
        
        if images is not None:
            min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
            shown_images = np.array([2 * proj.max(0)])
            for i in range(data.shape[0]):
                dist = np.sum((proj[i] - shown_images) ** 2, 1)
                if np.min(dist) < min_dist_2:
                    # On ne montre pas le points trop proches
                    continue
                shown_images = np.vstack([shown_images, proj[i]])
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(images[i], cmap=cmap),
                                        proj[i])
                ax.add_artist(imagebox)

    # fig, ax = plt.subplots(figsize=(10, 10))
    # plot_components(data = dataLLE, model = lle, 
    #                 images=data.values.reshape((-1, 28, 28)), 
    #                 ax=ax, thumb_frac=0.05, prefit = True)
    # fig.show()

    # On projette les images sur les 2 axes principaux de l'Isomap
    # isomap = Isomap(n_neighbors=50, n_components=2)
    # dataISO = isomap.fit_transform(data)

    # fig = plt.figure()

    # ax = fig.add_subplot(111)
    # ax.scatter(dataISO[:, 0], dataISO[:, 1],  c = target, cmap=plt.cm.Spectral, alpha = .7, s = 4)

    # ax.set_title("Données projetées sur les 2 composantes de Isomap")
    # plt.show()


    # tsne = TSNE(n_components=2, method = 'barnes_hut')
    # dataTSNE = tsne.fit_transform(data)

    # fig = plt.figure()

    # ax = fig.add_subplot(111)
    # ax.scatter(dataTSNE[:, 0], dataTSNE[:, 1],  c = target, cmap=plt.cm.Spectral, alpha = .7, s = 4)

    # ax.set_title("Données projetées sur les 2 axes de Tsne")
    # plt.show()


    # data_mant = data[target == 4]
    # # display(data_mant.head())
    # fig, ax = plt.subplots(figsize=(10, 10))
    # plot_components(data_mant, tsne, images=data_mant.values.reshape((-1, 28, 28)),
    #                 ax=ax, thumb_frac=0.1, cmap='gray_r')
    # fig.show()

    umap_model = umap.UMAP(n_neighbors = 15, min_dist = 0.1, n_components=2)
    embedding = umap_model.fit_transform(data)

    # Plot après UMAP
    plt.figure(figsize = (10,8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c = target)
    plt.title("Après UMAP")
    plt.show()


    mapper = umap.UMAP(n_neighbors = 15, min_dist = 0.1, n_components = 2).fit(data)
    umap.plot.points(mapper, labels=target)
    umap.plot.plt.show() 
    
    digits = load_digits()

    embedding = umap.UMAP().fit_transform(digits.data)
    embedding_bis =  umap.UMAP(n_neighbors=5,
                        min_dist=0.3,
                        metric='correlation').fit_transform(digits.data)

    mapper = umap.UMAP().fit(digits.data)
    umap.plot.points(mapper, labels=digits.target)
    umap.plot.plt.show() 

    # Pour bloquer la fenêtre d'affichage
    input('Appuyez Entrée pour quitter')

except FileNotFoundError:
    print("❌ Erreur: Le fichier 'data/AirPassengers.csv' n'a pas été trouvé.")
    print("Vérifiez que le fichier existe dans le dossier 'data'.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    sys.exit(1)
