import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from IPython.display import display
from matplotlib import offsetbox
from matplotlib.image import imread
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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

    # Séparation des features et de la target
    df = pd.concat([df_train, df_test], axis=0)
    target = df['label']
    data = df.drop('label',axis=1)

    # Superposition des images sur les 2 premières composantes principales
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

    # Supposons que le DataFrame df contient les images sous forme de données plates
    img = df.iloc[0, 1:].values  # Récupérer la première ligne et la transformer en array
    img = img.reshape((28, 28))  # Format 
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    plt.show()

    pca = PCA(n_components = 2)

    data_2D = pca.fit_transform(data)
    print(f"Variance expliquée par les 2 premières composantes: {sum(pca.explained_variance_ratio_)*100:.2f}%")
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.scatter(data_2D[:, 0], data_2D[:, 1], c = target, cmap=plt.cm.Spectral)

    ax.set_xlabel('PC 0')
    ax.set_ylabel('PC 1')

    ax.set_title("Données projetées sur les 2 axes de PCA")
    plt.show()

    pca3D = PCA(n_components = 3)
    data_3D = pca3D.fit_transform(data)
    print(f"Variance expliquée par les 3 premières composantes: {sum(pca3D.explained_variance_ratio_)*100:.2f}%")

    total_var = pca3D.explained_variance_ratio_.sum() * 100

    # matplotlib 3D
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(data_3D[:, 0], data_3D[:, 1], data_3D[:, 2],
    c=target, cmap=plt.cm.Spectral, s=20, depthshade=True)

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.set_title(f'PCA (3D) — Total variance: {total_var:.2f}%')

    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    plt.tight_layout()
    plt.show()

    # plotly 3D
    fig = px.scatter_3d(
    data_3D, x=0, y=1, z=2, color=target,
    title=f'PCA 3D — Total explained variance: {total_var:.2f}%',
    labels={'0':'PC 1', '1':'PC 2', '2':'PC 3'}
    )
    fig.show()

    # PCA sans composantes fixes
    pca_noc = PCA()
    pca_noc.fit(data)

    plt.figure()
    plt.xlim(0,100)
    plt.plot(pca_noc.explained_variance_ratio_)

    plt.figure()
    plt.xlim(0,100)
    plt.xlabel('Nombre de composantes')
    plt.ylabel('Part de variance expliquée')
    plt.axhline(y = 0.9, color ='r', linestyle = '--')
    plt.plot(pca_noc.explained_variance_ratio_.cumsum())
    plt.show()

    # PCA avec 90% de variance expliquée
    pca_wc = PCA(n_components = 0.9)
    pca_wc.fit(data)
    print("Nombre de composantes retenues :", pca_wc.n_components_)

    # Nouvel entrainement avec PCA
    y_test = df_test['label']
    y_train = df_train['label']

    X_test = df_test.drop('label', axis = 1)
    X_train = df_train.drop('label', axis = 1)

    # RandomForest sans PCA
    print("\n🔍 RandomForest sans PCA")
    rf_cls = RandomForestClassifier(n_jobs = -1)
    # L'argument n_jobs ne vaut pas -1 par défaut. Cette valeur permet de forcer le processeur à utiliser toute sa puissance de calcul parallèle.
    rf_cls.fit(X_train, y_train)
    print(rf_cls.score(X_test, y_test))

    pca = PCA(n_components = 0.9)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"Nombre de composantes PCA: {pca.n_components_}")
    print(f"Variance expliquée: {sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    print("\n🔍 RandomForest avec PCA")
    rf_cls = RandomForestClassifier(n_jobs = -1)
    # L'argument n_jobs ne vaut pas -1 par défaut. Cette valeur permet de forcer le processeur à utiliser toute sa puissance de calcul parallèle.
    rf_cls.fit(X_train_pca , y_train)
    print(rf_cls.score(X_test_pca, y_test))

    # LDA
    lda = LDA()
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    print(X_train_lda.shape)

    print("\n🔍 RandomForest avec LDA")
    rf_cls = RandomForestClassifier(n_jobs = -1)
    # L'argument n_jobs ne vaut pas -1 par défaut. Cette valeur permet de forcer le processeur à utiliser toute sa puissance de calcul parallèle.
    rf_cls.fit(X_train_lda , y_train)
    print(rf_cls.score(X_test_lda, y_test))
          
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1],  c = y_train, cmap=plt.cm.Spectral)

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')

    ax.set_title("Données projetées sur les 2 axes de PCA")
    plt.show();

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c = y_train, cmap=plt.cm.Spectral)

    ax.set_xlabel('LD 1')
    ax.set_ylabel('LD 2')

    ax.set_title("Données projetées sur les 2 axes de LDA")
    plt.show()


    # Pour bloquer la fenêtre d'affichage
    input('Appuyez Entrée pour quitter')

except FileNotFoundError:
    print("❌ Erreur: Le fichier 'data/AirPassengers.csv' n'a pas été trouvé.")
    print("Vérifiez que le fichier existe dans le dossier 'data'.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    sys.exit(1)
