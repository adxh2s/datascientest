# model learning regression lineaire
from sklearn import linear_model, preprocessing 
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Pour l'encodage des variables catégorielles
from sklearn.preprocessing import OneHotEncoder

# Pandas et Numpy pour dataframe et calcul
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

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
        
        
# chargement donnnées
df = pd.read_csv('../data/winequality-red.csv') #, index_col=0)

# Informations du dataframe
display_dataframe_info(df)

# Remplacer les espaces par des underscores dans les noms de colonnes
df.columns = [col.replace(' ', '_') for col in df.columns]
    
# separation des données et de la cible
# On suppose que la colonne 'quality' est la cible
data = df.drop(columns='quality', axis=1)
target = df['quality']

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=66)

# Standardisation des données d'entraînement
X_train_scaled = preprocessing.scale(X_train)

# Moyenne et écart-type des données d'entraînement
print(X_train_scaled.mean(axis=0))
print(X_train_scaled.std(axis=0))

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

print(X_train_scaled.mean(axis=0))
print(X_train_scaled.std(axis=0))

# Standardisation des données de test
X_test_scaled = scaler.transform(X_test)

print(X_test_scaled.mean(axis=0))
print(X_test_scaled.std(axis=0))

# Instanciation de la classe SVM avec un noyau polynomial
from sklearn import svm
clf = svm.SVC(gamma=0.01, kernel='poly')
# Entraînement du modèle SVM
# On utilise les données d'entraînement standardisées
clf.fit(X_train_scaled, y_train)

# Prédiction sur les données de test
# On utilise les données de test standardisées
y_pred = clf.predict(X_test_scaled)

# Methode 1 : Affichage de la matrice de confusion
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de confusion")
plt.show()
# Méthode 2 : Affichage de la matrice de confusion avec pandas
cm2 = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
print(cm2)

# On va passer par une grille de modèles pour évaluer les parametres
parametres = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear', 'poly'],
    'gamma': [0.001, 0.1, 0.5],
}
# Instanciation de la grille de recherche
grid_clf = GridSearchCV(estimator=clf, param_grid=parametres)
# Entraînement de la grille de recherche
# On utilise les données d'entraînement standardisées
grille = grid_clf.fit(X_train_scaled, y_train)
# Affichage des résultats de la grille de recherche
print("Meilleurs paramètres trouvés:")
print(pd.DataFrame.from_dict(grille.cv_results_).loc[:,['params', 'mean_test_score']]) 

# Affichage des meilleurs paramètres
print("Meilleurs paramètres:")
print(grid_clf.best_params_)

# Prédiction avec les meilleurs paramètres
# On utilise les données de test standardisées
y_pred = grid_clf.predict(X_test_scaled)
# Affichage de la matrice de confusion avec les meilleurs paramètres
cm3 = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
print(cm3)

# Comparaison du score du modèle avec les meilleurs paramètres
linear_train_sizes = [50, 70, 80, 100, 110, 118]
linear_train_sizes, linear_train_scores, linear_valid_scores = learning_curve(svm.SVC(kernel='linear', C= 1), data, target, train_sizes=linear_train_sizes, cv=5)

# Partie linear
train_scores_mean = np.mean(linear_train_scores, axis=1)
train_scores_std = np.std(linear_train_scores, axis=1)
valid_scores_mean = np.mean(linear_valid_scores, axis=1)
valid_scores_std = np.std(linear_valid_scores, axis=1)

# figure pour la courbe d'apprentissage
fig = plt.figure(figsize=(10, 10))
axis = fig.add_subplot(121)
# Grille
axis.grid()

# Halo autour de la courbe (bandes d'écart-type)
axis.fill_between(linear_train_sizes, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.1, color="r")
axis.fill_between(linear_train_sizes, valid_scores_mean - valid_scores_std,
                valid_scores_mean + valid_scores_std, alpha=0.1, color="g")

# Courbes
axis.plot(linear_train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
axis.plot(linear_train_sizes, valid_scores_mean, 'o-', color="g", label="Cross-validation score")

# Légende et titre
axis.legend(loc="best")
axis.set_title("Courbe d'apprentissage svm linear")
axis.set_xlabel("Training examples")
axis.set_ylabel("Score")


# Affichage de la courbe d'apprentissage
grid_train_sizes = [50, 70, 80, 100, 110, 118]
grid_train_sizes, grid_train_scores, grid_test_scores = learning_curve(grid_clf, data, target, n_jobs=4, train_sizes=grid_train_sizes)

# Partie grid
train_scores_mean = np.mean(grid_train_scores, axis=1)
train_scores_std = np.std(grid_train_scores, axis=1)
test_scores_mean = np.mean(grid_test_scores, axis=1)
test_scores_std = np.std(grid_test_scores, axis=1)

# figure pour la courbe d'apprentissage
axis = fig.add_subplot(122)
# Grille
axis.grid()
# Halo autour des courbes (bandes d’incertitude)
axis.fill_between(grid_train_sizes, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.1, color="r")
axis.fill_between(grid_train_sizes, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha=0.1, color="g")

# Courbes des scores moyens d’entraînement et de validation croisée
axis.plot(grid_train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
axis.plot(grid_train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

# Légende, titre et labels des axes
axis.legend(loc="best")
axis.set_title("Courbe d'apprentissage svm avec grille de recherche")
axis.set_xlabel("Training examples")
axis.set_ylabel("Score")

plt.show()