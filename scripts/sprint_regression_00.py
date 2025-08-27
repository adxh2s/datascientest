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
df = pd.read_csv("../data/2023_nba_player_stats.csv")

# Afficher les informations du DataFrame
display_dataframe_info(df)

# On renomme les colonnes pour faciliter l'accès
# df = df.rename(
#     columns={"Pname": "player", "Team": "bref_team_id", "POS": "pos",
#              "PTS": "pts"}
# )
# display(df.head())

# On re-définit l'index de notre dataframe nba avec le joueur et l'id de l'équipe
# df.index = df['player'] + ' - ' + df['bref_team_id']
df.index = df['PName'] + ' - ' + df['Team']

# # On supprime les données manquantes
df = df.dropna()

# # On affiche le nombre d'occurences de chaque poste
display(df['POS'].value_counts())

# # On assigne la variable pos à pl_pos
pl_pos = df['POS']

# # On supprime les colonnes season, player, bref_team_id et pos
df = df.drop(['PName', 'Team', 'POS'], axis=1)

# # On ajoute au dataframe les dummies
df = df.join(pd.get_dummies(pl_pos, prefix='pos'))

# Nous pouvons supprimer une des colonnes relatives au poste car elle n'apporte aucune information supplémentaire. En effet,
# Si un joueur n'est à aucun des postes présents dans le tableau, on peut en déduire qu'il détient le dernier poste que nous
# avons retiré.
# df = df.drop('pos_SG', axis=1)

# Préparation des données pour la régression
# On sépare les variables explicatives et la variable cible
X = df.drop("PTS", axis=1)
y = df["PTS"]

# On sépare les données en ensembles d'entraînement et de test
# On utilise 80% des données pour l'entraînement et 20% pour le test
# la bonne pratique est de fixer un random_state pour que les résultats soient reproductibles
# on doit normalement diviser les données en 3 ensembles : train, validation et test
# mais ici nous n'avons pas de validation, donc nous allons utiliser la validation croisée
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101
)

# Normalisation des données sur  l'ensemble d'entraînement, puis application de la même transformation sur l'ensemble de test
# On utilise StandardScaler pour normaliser les données
scaler = preprocessing.StandardScaler()
# On ajuste le scaler sur l'ensemble d'entraînement et on transforme les données
X_train[X_train.columns] = pd.DataFrame(
    scaler.fit_transform(X_train), index=X_train.index
)
# On transforme l'ensemble de test avec le même scaler
X_test[X_test.columns] = pd.DataFrame(
    scaler.transform(X_test), index=X_test.index
)

# Affichage de la heatmap des corrélations
# préparation de la figure
plt.figure(figsize=(16, 15))
# heatmap des corrélations du dataframe nba df
sns.heatmap(df.corr(), annot=True, cmap="RdBu_r", center=0)
plt.title("Heatmap des corrélations des variables du dataframe NBA")
plt.show()

# Régression linéaire simple
# On crée un modèle de régression linéaire
lr1 = LinearRegression()
# On entraîne le modèle sur l'ensemble d'entraînement, colonne 'mp' (minutes jouées) comme variable explicative
# et 'pts' (points marqués) comme variable cible
lr1.fit(X_train[["Min"]], y_train)

#  Le coefficient de détermination (score  𝑅2)
# --> Plus le coefficient de détermination est proche de 1, plus le modèle s'ajuste bien aux données.
#  La racine de l'erreur quadratique moyenne (RMSE)
# --> Plus la RMSE est faible, plus les prédictions sont bonnes.
# Nous regardons ces métriques sur l'ensemble train et sur l'ensemble test
# afin de voir les performances du modèle sur des données qu'il connait déjà et sur des données qu'il n'a jamais vues.
print(
    "score R² (coefficient de détermination) de l'ensemble train :",
    lr1.score(X_train[["Min"]], y_train),
)
print(
    "score R² (coefficient de détermination) de l'ensemble test  :",
    lr1.score(X_test[["Min"]], y_test),
)

# On prédit les points marqués sur l'ensemble d'entraînement et de test
pred = lr1.predict(X_train[["Min"]])  # ou y_pred
pred_test = lr1.predict(X_test[["Min"]])  # ou y_pred_test
print("rmse train :", np.sqrt(mean_squared_error(y_train, pred)))
print("rmse test : ", np.sqrt(mean_squared_error(y_test, pred_test)))

# Régression linéaire multiple
# Modèle de régression linéaire multiple ElasticNetCV (Cross-Validation)
# cv=8 signifie que nous utilisons 8 plis pour la validation croisée
# l1_ratio est un paramètre qui contrôle le mélange entre Lasso (l1) et Ridge (l2)
# alphas est une liste de valeurs pour le paramètre alpha, qui contrôle la régularisation
model_en = ElasticNetCV(
    cv=8,
    l1_ratio=(0.1, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.99),
    alphas=(0.001, 0.01, 0.02, 0.025, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0),
)
# On entraîne le modèle sur l'ensemble d'entraînement
model_en.fit(X_train, y_train)

# On prépare un graphique pour visualiser les performances du modèle

# On récupère les alphas utilisés dans le modèle
alphas = model_en.alphas_

# Préparation de la figure
plt.figure(figsize=(10, 10))

# On trace la courbe de l'erreur quadratique moyenne pour chaque alpha
# model_en.mse_path_ est un tableau 3D, on prend la moyenne sur la première dimension (l1_ratio)
# et on trace la moyenne pour chaque alpha
for i in range(model_en.mse_path_.shape[0]):
    plt.plot(
        alphas,
        model_en.mse_path_[i, :, :].mean(axis=1),
        label="Moyenne pour l1_ratio= %.2f" % model_en.l1_ratio[i],
        linewidth=2,
    )

# Légende et les labels
plt.xlabel("Alpha")
plt.ylabel("Mean squared error")
plt.title("Mean squared error pour chaque $\lambda$")
plt.legend()
plt.show()

# Affichage des coefficients du modèle
print(
    "score R² (coefficient de détermination) de l'ensemble train :",
    model_en.score(X_train, y_train),
)
print(
    "score R² (coefficient de détermination) de l'ensemble test  :",
    model_en.score(X_test, y_test),
)

# On prédit les points marqués sur l'ensemble d'entraînement et de test
y_pred_train = model_en.predict(X_train)
y_pred_test = model_en.predict(X_test)
# Erreur quadratique moyenne (RMSE) pour l'ensemble d'entraînement et de test
print("rmse train :", np.sqrt(mean_squared_error(y_train, y_pred_train)))
print("rmse test :", np.sqrt(mean_squared_error(y_test, y_pred_test)))

# On compare les points observés et les points prédits
moy = scaler.mean_[-1]
ec = scaler.scale_[-1]
print("moyenne :", moy)
print("ecart-type :", ec)

# On affiche les 7 premières lignes du DataFrame avec les points observés et les points prédits
display(
    pd.DataFrame(
        {"points_observés": y_test, "points_predits": pred_test},
        index=X_test.index,
    ).head(10)
)
