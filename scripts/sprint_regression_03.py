import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error

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

# standardisation des données
# V1 tranformation directe (uniquement sur de l'exploration)
# scaler = preprocessing.StandardScaler()
# X_train[X_train.columns] = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index)
# X_test[X_test.columns] = pd.DataFrame(scaler.transform(X_test), index=X_test.index)
# ------------------------------------------
# V2 standardisation des données


# version propre pour la prod
def scale_dataframe(scaler, df):
    return pd.DataFrame(
        scaler.transform(df),
        columns=df.columns,
        index=df.index
    )


# Standardisation des données
# Créer un scaler et l'ajuster sur les données d'entraînement
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
# Appliquer la transformation sur les données d'entraînement et de test
X_train_scaled = scale_dataframe(scaler, X_train)
X_test_scaled = scale_dataframe(scaler, X_test)

# Ridge Regression avec validation croisée
from sklearn.linear_model import RidgeCV

# instanciation d'un RidgeCV
rcv = RidgeCV(alphas=(0.001, 0.01, 0.1, 0.3, 0.7, 1, 10, 50, 100))
# Entraînement sur les données d'apprentissage standardisées
rcv.fit(X_train_scaled, y_train)

# alpha retenu
print('alpha sélectionné par c-v :', rcv.alpha_)

# score R² entraînement et test
print( "R² - Coefficient de détermination du modèle ridge sur train     :", rcv.score(X_train_scaled, y_train))
print( "R² - Coefficient de détermination du modèle ridge sur test      :", rcv.score(X_test_scaled, y_test))

# Valeurs ajustées du modèle ou prédictions
rcv_pred_train = rcv.predict(X_train_scaled)
rcv_pred_test = rcv.predict(X_test_scaled)

# affichage MSE
print('MSE - Erreur quadratique moyenne de prédiction sur train: ', mean_squared_error(rcv_pred_train, y_train))
print('MSE - Erreur quadratique moyenne de prédiction sur test : ', mean_squared_error(rcv_pred_test, y_test))

# Modèle Lasso
from sklearn.linear_model import Lasso

# Entraînement du modèle Lasso avec alpha=1
lasso_r = Lasso(alpha=1)
lasso_r.fit(X_train_scaled, y_train)

# Affichage des coefficients
print("Coefficients de la régression linéaire simple :")
print("ordonnée à l'origine : ", lasso_r.intercept_)
for index, coef in enumerate(lasso_r.coef_):
    print("variable", lasso_r.feature_names_in_[index], "- pente ou coefficient estimé : ", coef)
    
# OU Extraire les coefficients et l'intercept
coeffs = list(lasso_r.coef_)
coeffs.insert(0, lasso_r.intercept_)
# Ajouter le nom de l'intercept et les noms des colonnes
feats = list(X.columns)
feats.insert(0, "intercept")
# Afficher les coefficients dans un DataFrame
display(pd.DataFrame({"valeur estimée": coeffs}, index=feats))

# Entraînement du modèle Lasso avec alpha=10
lasso_r2 = Lasso(alpha=10)
lasso_r2.fit(X_train_scaled, y_train)
# Affichage des coefficients
plt.plot(range(len(X.columns)), lasso_r2.coef_)
plt.xticks(range(len(X.columns)), X.columns.values, rotation=70)
plt.xlabel('Variables')
plt.ylabel('Coefficients')
plt.title('Coefficients du modèle Lasso avec alpha=10')
plt.grid()
plt.show()

# score R² entraînement et test
print( "R² - Coefficient de détermination du modèle Lasso alpha 10 sur train: ", lasso_r2.score(X_train_scaled, y_train))
print( "R² - Coefficient de détermination du modèle Lasso alpha 10 sur test : ", lasso_r2.score(X_test_scaled, y_test))

# Valeurs ajustées du modèle ou prédictions
lasso_r2_pred_train = lasso_r2.predict(X_train_scaled)
lasso_r2_pred_test = lasso_r2.predict(X_test_scaled)

# affichage MSE
print('MSE - Erreur quadratique moyenne de prédiction Lasso alpha 10 sur train: ', mean_squared_error(lasso_r2_pred_train, y_train))
print('MSE - Erreur quadratique moyenne de prédiction Lasso alpha 10 sur test : ', mean_squared_error(lasso_r2_pred_test, y_test))

# Lasso avec validation croisée pour trouver le meilleur alpha
from sklearn.linear_model import LassoCV

# Entraînement du modèle Lasso avec validation croisée
lcv = LassoCV(cv=10)
lcv.fit(X_train_scaled, y_train)
print('Valeur alpha optimale :', lcv.alpha_)

# Affichage des coefficients
plt.figure(figsize=(10, 8))
plt.plot(lcv.alphas_, lcv.mse_path_, ':')
plt.plot(lcv.alphas_, lcv.mse_path_.mean(axis=1), 'k', label='moyenne des MSE')
plt.axvline(lcv.alpha_, linestyle='--', color='k', label='alpha : estimation CV')
plt.legend()
plt.xlabel('Alpha')
plt.ylabel('Mean square error')
plt.title('Mean square error pour chaque échantillon')
plt.show()

# affichage R²
print('score train lcv :', lcv.score(X_train_scaled, y_train))
print('score test lcv  :', lcv.score(X_test_scaled, y_test))

# Valeurs ajustées du modèle ou prédictions
lcv_pred_train = lcv.predict(X_train_scaled)
lcv_pred_test = lcv.predict(X_test_scaled)

# affichage MSE
print('Erreur quadratique moyenne de prédiction lcv sur train: ', mean_squared_error(lcv_pred_train, y_train))
print('Erreur quadratique moyenne de prédiction lcv sur test : ', mean_squared_error(lcv_pred_test, y_test))
