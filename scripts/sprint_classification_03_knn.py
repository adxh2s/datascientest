# model learning regression lineaire
from sklearn import svm, preprocessing, model_selection, neighbors, datasets 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
        
        
# chargement donnnées dans un dictionnaire
# On utilise le dataset digits de sklearn pour la classification
dict_df = datasets.load_digits()
for key, value in enumerate(dict_df):
    print("key : ", key)
# Sélection des données et de la cible
X = pd.DataFrame(dict_df['data'])
y = dict_df['target']

# Informations du dataframe
display_dataframe_info(X, title="Aperçu des données X")

# affichage 
j=0

for i in np.random.choice(np.arange(0, len(y)), size=6):
    j=j+1
#On stocke l'indice dans la liste i pour pouvoir afficher le label correspondant plus tard.
    
    plt.subplot(2,3,j)
# Rajouter *plt.subplot(2,3,j)* à chaque itération permet d'afficher toutes les images
# ensembles sur la même figure.

    plt.axis('off')
# Permet de supprimer les axes (ici sert à mieux voir les titres)
    
    plt.imshow(dict_df.images[i], cmap = cm.binary, interpolation='None')
# Affiche l'image n°i
# L'utilisation de cm.binary permet de voir les chiffres en gris sur fond blanc.

    plt.title('Label: %i' %y[i])
# Pour chaque image on écrit en titre le label qui lui correspond

plt.show()

# Décomposition des données en deux ensembles d'entraînement et de test
# par défaut l'échantillon est aléatoirement réparti
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=126)

# help(neighbors.KNeighborsClassifier)
# choix du modèle
knn = neighbors.KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2)
# Ajustement du modèle aux données d'entraînement
knn.fit(X_train, y_train)

# Prédiction sur les données de test
# On utilise la méthode predict() pour prédire les classes des données de test
y_pred = knn.predict(X=X_test)
# Affichage de la matrice de confusion
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot(cmap=plt.cm.Blues)
# Affichage de la matrice de confusion
plt.title("Matrice de confusion")
plt.show()
# ou via pandas
cm2 = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
print(cm2)

# Création du classifieur et construction du modèle sur les données d'entraînement
knn_m = neighbors.KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn_m.fit(X_train, y_train)
# Score des 2 modèles
print(knn.score(X_test, y_test))
print(knn_m.score(X_test, y_test))

# Comparaison des scores pour différentes métriques et valeurs de k voisins
score_minko = []
score_man = []
score_cheb = []

for k in range(1, 41):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score_minko.append(knn.score(X_test, y_test))

for k in range(1, 41):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    knn.fit(X_train, y_train)
    score_man.append(knn.score(X_test, y_test))
    
for k in range(1, 41):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k, metric='chebyshev')
    knn.fit(X_train, y_train)
    score_cheb.append(knn.score(X_test, y_test))
    
# Affichage des scores pour chaque métrique
plt.figure(figsize=(12, 6))    
plt.grid(visible=True)
plt.plot(range(1,41), score_minko, 'r', label='min')
plt.plot(range(1,41), score_man, 'b', label='man')
plt.plot(range(1,41), score_cheb, 'y', label='cheb')
plt.legend(loc='best')
plt.xlabel('k')
plt.ylabel('score')
plt.show()