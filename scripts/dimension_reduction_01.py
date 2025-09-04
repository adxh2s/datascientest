import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import display
from sklearn.feature_selection import (
    RFE,
    RFECV,
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_regression,
    mutual_info_regression,
)
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler

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
    df  = pd.read_csv('data/real_estate.csv')
    display(df.info())
    display(df.describe())
    print("✅ Données chargées avec succès!")
    
    # Séparation des données en train et test    
    X_train, X_test, y_train, y_test = train_test_split(df.drop('MEDV', axis = 1),df['MEDV'],test_size = .2)

    # Methodes de sélection de features
    # Méthodes de filtrage
    print("\n🔍 Sélection de features - Méthodes de filtrage")
    
    # 1. VarianceThreshold
    print("1. VarianceThreshold")
    filtering_selector = VarianceThreshold(threshold=1e-06)
    filtering_selector.fit(X_train)

    # On récupère le masque des features sélectionnées
    mask = filtering_selector.get_support()
    # On transforme en matrice 2D pour affichage
    M = mask.reshape(1, -1)
    # Affichage
    fig, ax = plt.subplots(figsize=(12, 2.5))
    im = ax.matshow(M, cmap = 'gray_r')

    # complément pour affichage
    # x_samples, x_features = X_train.shape
    # colnames = [f"feat_{i}" for i in range(x_features)]
    # ou vrais noms des colonnes
    colnames = list(X_train.columns)
    print(colnames)

    m_samples, m_features = M.shape

    print(m_features)

    for x in range(m_features + 1):
        ax.axvline(x - 0.5, color='lightgray', linewidth=0.8, zorder=0)

    # Ticks et labels en haut, rotation 22.5°
    ax.set_xticks(range(m_features))
    ax.set_xticklabels(colnames, rotation=90, ha='right')
    #, rotation_mode='anchor')
    ax.xaxis.set_ticks_position('top')

    # Cacher l'axe Y (une seule ligne)
    ax.set_yticks([])
    ax.set_xlabel('Feature selection')

    # legende
    cmap = plt.get_cmap('gray_r') # même cmap que matshow
    color_ignored = cmap(0.0) # valeur 0 -> sombre (ignored)
    color_selected = cmap(1.0) # valeur 1 -> clair (selected)

    legend_patches = [
    Patch(facecolor=color_selected, edgecolor='black', label='feature selected'),
    Patch(facecolor=color_ignored, edgecolor='black', label='feature ignored'),
    ]
    leg = ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=2, frameon=True)

    # affichage
    plt.tight_layout()
    plt.show()


    X_train_save = X_train
    X_test_save = X_test
    X_train = sel.transform(X_train)
    X_test = sel.transform(X_test)

    print(f"Nombre de features après: {X_train.shape[1]}")
    print()


    # Pour bloquer la fenêtre d'affichage
    input('Appuyez Entrée pour quitter')

except FileNotFoundError:
    print("❌ Erreur: Le fichier 'data/AirPassengers.csv' n'a pas été trouvé.")
    print("Vérifiez que le fichier existe dans le dossier 'data'.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    sys.exit(1)
