import os
import sys

# Pour éviter d'avoir les messages warning
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
from IPython.display import display
from matplotlib import offsetbox
from matplotlib.image import imread

# Configuration pour un affichage plus riche
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)


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
    df = pd.read_csv('data/breast_cancer.csv')
    display(df.head())
    print("✅ Données chargées avec succès!")
    print()
    display_dataframe_info(df, title="Breast Cancer Dataset Overview")

    # Changer les valeurs de la colonne 'diagnosis' pour plus de clarté
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    # Changer le type de la colonne 'diagnosis' en int
    df['diagnosis'] = df['diagnosis'].astype("Int64")

    df = df.drop(columns=['id', 'Unnamed: 32'])

    # Afficher la matrice de corrélation
    plt.figure(figsize=(20, 20))  # Taille de la figure
    sns.heatmap(df.corr(), annot=True, cmap='viridis');
    plt.show()

    # La variable cible : 'diagnosis'
    target = df['diagnosis']

    # Supression de 'diagnosis'
    df.drop('diagnosis', axis=1, inplace=True)

    sc = StandardScaler()
    Z = sc.fit_transform(df)
    display(Z.shape)

    pca = PCA()
    # Contient les coordonnées de l'ACP sur les lignes.
    coord_pca = pca.fit_transform(Z)
    
    print(f"Variance expliquée par chaque composante :\n{pca.explained_variance_ratio_}")
    print(f"\nVariance totale expliquée : {pca.explained_variance_ratio_.sum()}")
    print(f"\nVariance cumulée expliquée : {pca.explained_variance_ratio_.cumsum()}")
    # Affichage des deux premières composantes principales
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=coord_pca[:, 0], y=coord_pca[:, 1], hue=target, palette=['green', 'red'])
    plt.title('Projection des données sur les deux premières composantes principales')
    plt.xlabel('Première composante principale')
    plt.ylabel('Deuxième composante principale')
    plt.legend(title='Diagnostic', labels=['Bénin', 'Malin'])
    plt.show()  
    
    plt.figure()
    plt.xlim(0,30)
    plt.plot(pca.explained_variance_ratio_)
    plt.show()  

    plt.figure()
    plt.xlim(0,30)
    plt.xlabel('Nombre de composantes')
    plt.ylabel('Part de variance expliquée')
    plt.axhline(y = 0.9, color ='r', linestyle = '--')
    plt.plot(pca.explained_variance_ratio_.cumsum())
    plt.show()  

    # Camembert de la répartition de la part de variance expliquée par chaque axe.
    n_pca = 6
    L1 = list(pca.explained_variance_ratio_[0:n_pca])
    L1.append(sum(pca.explained_variance_ratio_[n_pca:31]))

    plt.pie(L1, labels=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'Autres'], 
            autopct='%1.3f%%');    
    plt.show()

    # Composantes principales et variances associées
    component_0 = pca.components_[0,:]
    explained_var_0 = pca.explained_variance_[0]

    component_1 = pca.components_[1,:]
    explained_var_1 = pca.explained_variance_[1]

    # Corrélations entre les variables initiales et les deux premières composantes principales
    corr_axe0 = component_0 * np.sqrt(explained_var_0)
    display(corr_axe0)

    corr_axe1 = component_1 * np.sqrt(explained_var_1)
    display(corr_axe1)

    # Tableau des charges factorielles
    charges_factorielles = pd.DataFrame(
    [corr_axe0, corr_axe1],
    columns=df.columns,
    index=['Axe 0', 'Axe 1'])

    display(charges_factorielles)
    # Cercle des corrélations
    def draw_correlation_circle(df_charges_factorielles, pca, arrow_length=0.01, label_rotation=0):
        fig, ax = plt.subplots(figsize=(8, 8))
        for i, var in enumerate(df_charges_factorielles.columns):
            x = df_charges_factorielles.loc['Axe 0', var]
            y = df_charges_factorielles.loc['Axe 1', var]
            ax.arrow(0, 0, x, y, head_width=arrow_length, head_length=arrow_length, fc='gray', ec='gray')
            ax.text(x, y, var,
                    ha='center', va='center',
                    fontsize=9, rotation=label_rotation, clip_on=True)
        circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='black')
        ax.add_artist(circle)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Axe 0 (PC0)')
        ax.set_ylabel('Axe 1 (PC1)')
        ax.set_title('Cercle des Corrélations')
        plt.grid()
        plt.show()
        

    # Appelez la fonction pour tracer le cercle de corrélation
    draw_correlation_circle(charges_factorielles, pca)

    # Préparation des données pour le graphique    
    df_plot = pd.DataFrame(coord_pca[:, :2], columns=["PC1", "PC2"])  
    df_plot["target"] = pd.Series(target).reset_index(drop=True)

    # Affichage du graphique
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        data=df_plot,
        x='PC1', y='PC2',
        hue='target',           # coloration selon la variable cible
        palette='Set2',         # ou 'viridis', 'coolwarm', etc.
        s=60, edgecolor='k'
    )
    plt.title("Projection PCA - composantes 1 et 2")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title='Cible')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Pour bloquer la fenêtre d'affichage
    input('Appuyez Entrée pour quitter')

except FileNotFoundError:
    print("❌ Erreur: Le fichier 'data/breast_cancer.csv' n'a pas été trouvé.")
    print("Vérifiez que le fichier existe dans le dossier 'data'.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    sys.exit(1)
