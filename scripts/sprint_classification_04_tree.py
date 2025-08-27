# model learning regression lineaire
from sklearn import svm, preprocessing, model_selection, neighbors, datasets 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from ucimlrepo import fetch_ucirepo

# Pandas et Numpy pour dataframe et calcul
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from IPython.display import display

import warnings

warnings.filterwarnings("ignore")

# Configuration pour un affichage plus riche
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 50)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", 50)


def display_dataframe_info(df, title="DataFrame Info"):
    """Affiche les informations du DataFrame de manière formatée"""
    print("=" * 80)
    print(f"📊 {title}")
    print("=" * 80)

    # Informations de base
    print(
        f"📋 Forme du DataFrame: {df.shape[0]} lignes × {df.shape[1]} colonnes"
    )
    print(
        f"💾 Utilisation mémoire: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    )
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
    categorical_cols = df.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 0:
        print("🏷️  Aperçu des valeurs uniques (colonnes catégorielles):")
        print("-" * 40)
        for col in categorical_cols[
            :5
        ]:  # Limite à 5 colonnes pour éviter l'encombrement
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

    if df[column_name].dtype == "object":
        print("\nValeurs les plus fréquentes:")
        print(df[column_name].value_counts().head(10))
    elif df[column_name].dtype in ["int64", "float64"]:
        print(f"\nMin: {df[column_name].min()}")
        print(f"Max: {df[column_name].max()}")
        print(f"Moyenne: {df[column_name].mean():.2f}")
        print(f"Médiane: {df[column_name].median():.2f}")
        
        
# Chargement des données
df = pd.read_csv('../data/breast_cancer_wisconsin.csv', index_col=0)

# Informations du dataframe
display_dataframe_info(df)

# séparation des données
data = df.drop(columns='Diagnosis', axis=1)
target = df['Diagnosis']

# Décomposition des données en deux ensembles d'entraînement et de test
# par défaut l'échantillon est aléatoirement réparti
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=123)

# Instanciation du classifier
dt_clf = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=123)

# Entraînement
dt_clf.fit(X_train, y_train)

# Application du modele sur l'ensemble de tests pour prédiction
y_pred = dt_clf.predict(X_test)

# Matrice de confusion
print(pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite']))

# Affichage des criteres de Tree selon l'importance des features
feats = {}

for name, importance in zip(dt_clf.feature_names_in_ , dt_clf.feature_importances_):
    feats[name] = importance
    
importances_entropy = pd.DataFrame.from_dict(feats, orient='index')

importances_entropy = importances_entropy.rename(columns={0: 'Importance'})

importances_entropy = importances_entropy['Importance'].sort_values(ascending=False)

display(importances_entropy.head(8))

# classifier type gini
dt_clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=321)

# Ajustement du modèle sur le jeu d'entraînement
dt_clf_gini.fit(X_train, y_train)

# prdiction sur le jeu de test
y_pred = dt_clf_gini.predict(X_test)

# Matrice de confusion
print(pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite']))

# Affichage des criteres de Tree selon l'importance des features
feats_gini = {}
for feature, importance in zip(data.columns, dt_clf_gini.feature_importances_):
    feats_gini[feature] = importance 
    
importances_gini = pd.DataFrame.from_dict(feats_gini, orient='index').rename(columns={0: 'Importance'})

# Affichage des 8 variables les plus importantes
display(importances_gini.sort_values(by='Importance', ascending=False).head(8))

# affichage des arbres de décision
feature_names = list(df.columns[:-1])  # récupérer les noms des features
plt.figure(figsize=(20, 10))
plt.title("Arbre de décision avec critère 'entropy'")
plot_tree(dt_clf, feature_names=feature_names, filled=True)
plt.show()

plt.figure(figsize=(20, 10))
plt.title("Arbre de décision avec critère 'gini'")
plot_tree(dt_clf_gini, feature_names=feature_names, filled=True)
plt.show()