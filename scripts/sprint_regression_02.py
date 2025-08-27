import numpy as np
import pandas as pd

from sklearn import model_selection, preprocessing
from sklearn.model_selection import (
    cross_val_predict,
    cross_val_score,
    cross_validate,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import scipy.stats as stats
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
df = pd.read_csv("../data/CarPrice_Assignment.csv", index_col=0)

# Afficher les informations du DataFrame
display_dataframe_info(df)

# Visualiser la relation entre 'curb-weight' et 'price'
df.rename(columns={"curbweight": "curb-weight"}, inplace=True)
# Afficher des informations spécifiques sur certaines colonnes
explore_column(df, "curb-weight")
explore_column(df, "price")

# Visualisation de la relation entre 'curb-weight' et 'price'
# plt.style.use('seaborn-darkgrid')
# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(111)
# ax.scatter(x=df['curb-weight'], y=df['price'])
# sns.relplot(x=df['curb-weight'], y=df['price'])
# plt.show()

# Sélectionner les colonnes pour X (data) et y (target)
# X = df[['curb-weight']] # renvoie un DataFrame pandas, equivalent à X = pd.DataFrame(df['curb-weight'])
# y = df['price'] # renvoie une série pandas
# display(df[['CarName', 'fueltype', 'aspiration', 'doornumber', 'carbody',
#             'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber',
#             'fuelsystem']].head())
X = df.drop(
    columns=[
        "CarName",
        "fueltype",
        "aspiration",
        "doornumber",
        "carbody",
        "drivewheel",
        "enginelocation",
        "enginetype",
        "cylindernumber",
        "fuelsystem",
        "price",
    ],
    axis="1",
)
y = df["price"]

# Séparation des données d'entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=789
)

# # standardisation des données
# scaler = preprocessing.StandardScaler()
# X_train[X_train.columns] = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index)
# X_test[X_test.columns] = pd.DataFrame(scaler.transform(X_test), index=X_test.index)

# Insérez votre code ici
slr = LinearRegression()

# Entrainement
slr.fit(X_train, y_train)

# Affichage des coefficients
print("Coefficients de la régression linéaire simple :")
# Extraire les coefficients et l'intercept
coeffs = list(slr.coef_)
coeffs.insert(0, slr.intercept_)
# Ajouter le nom de l'intercept et les noms des colonnes
feats = list(X.columns)
feats.insert(0, "intercept")
# Afficher les coefficients dans un DataFrame
display(pd.DataFrame({"valeur estimée": coeffs}, index=feats))

# Metriques R²
print(
    "Coefficient de détermination du modèle sur train     :",
    slr.score(X_train, y_train),
)
print(
    "Coefficient de détermination obtenu par Cv sur train :",
    cross_val_score(slr, X_train, y_train).mean(),
)
print(
    "Coefficient de détermination du modèle sur test      :",
    slr.score(X_test, y_test),
)

# prédiction à partir de l'environnement test
y_pred_test = slr.predict(X_test)
# Affichage des résultats de la prédiction
plt.scatter(y_pred_test, y_test)
plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), "r")
plt.show()

# prédiction à partir de l'environnement train
y_pred_train = slr.predict(X_train)
# Résidus sur l'environnement train
residus = y_pred_train - y_train
display(residus.describe())

# nuage des résidus
plt.scatter(y_train, residus, color="#980a10", s=15)
# ligne horizontale à 0
plt.axhline(y=0, color="blue", linestyle="--")
# OU plt.plot((y_train.min(), y_train.max()), (0, 0), lw=3, color='#0a5798')
plt.xlabel("Valeurs prédites depuis l'environnement train")
plt.ylabel("Résidus")
plt.title("Plot des résidus vs valeurs prédites (environnement train)")
plt.show()


# Normalité des résidus
import scipy.stats as stats

residus_norm = (residus - residus.mean()) / residus.std()
stats.probplot(residus_norm, plot=plt)
plt.show()

# Selection des colonnes pour la matrice de corrélation et le pairplot
cols = [
    "wheelbase",
    "carlength",
    "carwidth",
    "carheight",
    "curb-weight",
    "enginesize",
    "boreratio",
    "stroke",
    "compressionratio",
    "horsepower",
    "peakrpm",
    "citympg",
    "highwaympg",
    "price",
]
# Matrice de corrélation
# plt.figure(figsize=(16, 15))
# sns.heatmap(df[cols].corr(), annot=True, cmap="RdBu_r", center=0)
# plt.show()

# Pairplot pour visualiser les relations entre certaines variables
# sns.pairplot(data=df[['curb-weight', 'horsepower', 'highway-mpg', 'height', 'bore', 'width','price']])
# sns.pairplot(data=df[cols])
# plt.show()


# Pour vérification # signif_features = ['curb-weight', 'horsepower', 'boreratio', 'carwidth', 'price']
# Matrice de corrélation
# plt.figure(figsize=(16, 15))
# sns.heatmap(df[signif_features].corr(), annot=True, cmap="RdBu_r", center=0)
# plt.show()

# # Pairplot pour visualiser les relations entre certaines variables
# # sns.pairplot(data=df[['curb-weight', 'horsepower', 'highway-mpg', 'height', 'bore', 'width','price']])
# sns.pairplot(data=df[signif_features])
# plt.show()

# Sélection des caractéristiques significatives
signif_features = ['curb-weight', 'horsepower', 'boreratio', 'carwidth']

# Etape 2 --> Affinage du modele de regression multiple
# Entraînement du modèle avec les caractéristiques significatives
lr2 = LinearRegression()
lr2.fit(X_train[signif_features], y_train)

# Score (R²) sur l'entraînement
print('Score (R²) sur train :', lr2.score(X_train[signif_features], y_train))
# Score (R²) sur le test
print('Score (R²) sur test  :', lr2.score(X_test[signif_features], y_test))

# Sélection des meilleures caractéristiques avec SelectKBest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# Instancier SelectKBest avec f_regression
sk = SelectKBest(f_regression, k=4)
sk.fit(X_train, y_train)
display(X.columns[sk.get_support()])

# Transformation des données d'entraînement et de test
sk_train = sk.transform(X_train)
sk_test = sk.transform(X_test)

# Entraînement du modèle avec les données transformées
sklr = LinearRegression()
sklr.fit(sk_train, y_train)

print('Score (R²) sur train optimisé via SelectKBest:', sklr.score(sk_train, y_train))
print('Score (R²) sur test optimisé via SelectKBest :', sklr.score(sk_test, y_test))


# Utilisation de SelectFromModel pour la sélection de caractéristiques
from sklearn.feature_selection import SelectFromModel

# Regression Linéaire
lr4 = LinearRegression()

# Selection des meilleures colonnes par poids de prédiction
sfm = SelectFromModel(lr4)

# standardisation des données
scaler = preprocessing.StandardScaler().fit(X_train)

# format tableau (si dataframe --> pd.DataFrame()...)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrainement sur les données standardisées et sauvegarde des données transformées
sfm_train = sfm.fit_transform(X_train_scaled, y_train)
sfm_test = sfm.transform(X_test_scaled)

# Affichage des colonnes sélectionnées par SelectFromModel
display(X.columns[sfm.get_support()])

# Créer un modèle à partir des données sauvegardées
lr5 = LinearRegression()
lr5.fit(sfm_train, y_train)

# affichez le score du modèle sur les échantillons d'entraînement et de test
print('Score (R²) sur train normalisé et optimisé via SelectFromModel :', lr5.score(sfm_train, y_train))
print('Score (R²) sur test normalisé et optimisé via SelectFromModel  :', lr5.score(sfm_test, y_test))
