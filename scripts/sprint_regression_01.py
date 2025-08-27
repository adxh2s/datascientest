import numpy as np
import pandas as pd

from sklearn import model_selection, preprocessing

from sklearn.model_selection import (
    cross_val_predict,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.linear_model import (
    LinearRegression,
    LassoCV,
    RidgeCV,
    ElasticNetCV,
)
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

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


# Charger le dataset
df = pd.read_csv("../data/CarPrice_Assignment.csv")

# Afficher les informations du DataFrame
display_dataframe_info(df)

# Visualiser la relation entre 'curb-weight' et 'price'
df.rename(columns={'curbweight': 'curb-weight'}, inplace=True)
# Afficher des informations spécifiques sur certaines colonnes  
explore_column(df, 'curb-weight')
explore_column(df, 'price')

# Visualisation de la relation entre 'curb-weight' et 'price'
# plt.style.use('seaborn-darkgrid')
# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(111)
# ax.scatter(x=df['curb-weight'], y=df['price'])
sns.relplot(x=df['curb-weight'], y=df['price'])
plt.show()

# Sélectionner les colonnes pour X (data) et y (target)
X = df[['curb-weight']] # renvoie un DataFrame pandas, equivalent à X = pd.DataFrame(df['curb-weight'])
y = df['price'] # renvoie une série pandas

# Insérez votre code ici 
slr = LinearRegression()

# # Séparation des données d'entrainement et de test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

# # standardisation des données
# scaler = preprocessing.StandardScaler()
# X_train[X_train.columns] = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index)
# X_test[X_test.columns] = pd.DataFrame(scaler.transform(X_test), index=X_test.index)

# Entrainement
slr.fit(X, y)

# Metriques simples
# print("score R² (coefficient de détermination) de l'environnement train :",
#       slr.score(X_train, y_train))
# print("score R² (coefficient de détermination) de l'environnement test  :",
#       slr.score(X_test, y_test))

# prédiction
# y_pred_train = slr.predict(X_train)
# y_pred_test = slr.predict(X_test)
# print('rmse train :', np.sqrt(mean_squared_error(y_train, y_pred_train)))
# print('rmse test : ', np.sqrt(mean_squared_error(y_test, y_pred_test)))

# Affichage des coefficients
print("Coefficients de la régression linéaire simple :")
print("ordonnée à l'origine : ", slr.intercept_)
print("pente ou coefficient de la droite : ", slr.coef_)

# Validation croisée à 4 plis ou echantillons
scores = cross_validate(slr, X, y, return_train_score=True, cv=4)

for key in scores.keys():
    print(f"{key}: {scores[key]}")

# Moyennes
print("Moyenne test :", scores['test_score'].mean())
print("Moyenne entraînement :", scores['train_score'].mean())

# Prédiction variables ajustées et résidus
y_pred_prix = slr.predict(X)
residus = y_pred_prix - y
display(residus.describe())

# Graphique de régression
plt.figure(figsize=(10, 8))
plt.scatter(X, y, color='darkblue', label='Données réelles')
plt.plot(X, y_pred_prix, color='k', label='Régression linéaire simple X / y Prédit')
plt.legend()
plt.xlabel('Curb Weight')
plt.ylabel('Price')
plt.title('Régression Linéaire Simple : Curb Weight vs Price')
plt.show()

# Graphique des résidus
plt.scatter(y, residus, color='#980a10', s=15)
plt.plot((y.min(), y.max()), (0, 0), lw=3, color='#0a5798')
plt.show()

# Vérification de la normalité des résidus
import scipy.stats as stats
# normalisation des résidus
residus_norm = (residus-residus.mean())/residus.std()
# qq plot
stats.probplot(residus_norm, plot=plt)
plt.show()

# Histogramme des résidus
plt.hist(residus, bins=30)
plt.title('Histogramme des résidus')
plt.show()


# Importer la fonction f_regression pour la sélection de caractéristiques
from sklearn.feature_selection import f_regression

# Appliquer f_regression
f_scores, p_values = f_regression(X, y)

# Affichage
df_resultats = pd.DataFrame({
    'Feature': [f'X{i}' for i in range(X.shape[1])],
    'F-score': f_scores,
    'p-value': p_values
})

display(df_resultats)

# RMSE (Root Mean Squared Error)
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets)**2).mean())


# Calculer le RMSE pour les prédictions
print(rmse(y_pred_prix, y))

# Validation croisée avec prédictions
y_pred_prix_2 = cross_val_predict(slr, X, y, cv=4)

print(rmse(y_pred_prix_2, y))