# Import des librairies
import pandas as pd
import numpy as np

from scipy.stats import chi2_contingency

import statsmodels.formula.api as smf

from sklearn.preprocessing import (
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
    OneHotEncoder,
)

OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Affichage dans Jupyter Notebook
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


def plot_confusion_matrix_seaborn(cm, figsize=(6, 5), variable_name=None):
    """
    Affiche une matrice de confusion 2x2 avec fond coloré personnalisé et légende,
    uniquement avec Seaborn et Matplotlib.

    Paramètres :
    - cm : array-like 2x2 (ex: numpy array ou liste de listes)
        Matrice de confusion [[VN, FP], [FN, VP]]
    - figsize : tuple (largeur, hauteur)
        Taille de la figure matplotlib
    - variable_name : str ou None (optionnel)
        Nom de la variable concernée, affiché dans le titre

    Exemple d'utilisation :
    cm = np.array([[50, 5],
                   [3, 42]])
    plot_confusion_matrix_seaborn(cm, figsize=(7,6), variable_name="Classe cible")
    """
    # Labels classiques confusion matrix
    row_labels = ["Vrai Négatif", "Faux Négatif"]
    col_labels = ["Faux Positif", "Vrai Positif"]

    # Convertir en DataFrame pour seaborn (avec labels)
    df_cm = pd.DataFrame(cm, index=row_labels, columns=col_labels)

    # Couleurs personnalisées pour chaque catégorie
    # On crée une palette map par catégories pour colorer le fond via un DataFrame semblable
    # Vert clair pour VP/VN, rouge clair pour FP/FN
    colors_array = np.array([["#97DBAE", "#FF6978"], ["#FF6978", "#97DBAE"]])
    df_colors = pd.DataFrame(
        colors_array, index=row_labels, columns=col_labels
    )

    # Création de la figure
    plt.figure(figsize=figsize)

    # Heatmap avec cmap neutre et annot
    ax = sns.heatmap(
        df_cm,
        annot=True,
        fmt="d",
        cbar=False,
        linewidths=0.8,
        linecolor="black",
        square=True,
        cmap="Greys",
        mask=None,
        annot_kws={"weight": "bold", "size": 14},
    )

    # Colore le fond des cases individuellement avec les couleurs définies (semi-transparent)
    for y in range(df_cm.shape[0]):
        for x in range(df_cm.shape[1]):
            ax.add_patch(
                plt.Rectangle(
                    (x, y),
                    1,
                    1,
                    fill=True,
                    facecolor=df_colors.iat[y, x],
                    alpha=0.3,
                    edgecolor="none",
                    lw=0,
                )
            )

    # Construction du titre avec variable_name
    if variable_name:
        plt.title(
            f"Matrice de confusion pour la variable : {variable_name}",
            fontsize=16,
        )
    else:
        plt.title("Matrice de confusion", fontsize=16)

    # Légende
    legend_elements = [
        Patch(
            facecolor="#97DBAE",
            edgecolor="black",
            label="Vrais Positifs / Vrais Négatifs",
        ),
        Patch(
            facecolor="#FF6978",
            edgecolor="black",
            label="Faux Positifs / Faux Négatifs",
        ),
    ]
    plt.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.3, 1),
        title="Légende",
    )

    # Labels axes
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vérité terrain")

    plt.tight_layout()
    plt.show()


def display_contingency_table(df, col1, col2, normalize):
    """
    Affiche un tableau de contingence entre deux colonnes d'un DataFrame.

    :param df: DataFrame contenant les données
    :param col1: Nom de la première colonne
    :param col2: Nom de la deuxième colonne
    """

    print("Répartition de la variable: ", end="\n\n")
    print(df[col1].value_counts(normalize=True, dropna=False), end="\n\n")
    print(df[col2].value_counts(normalize=True, dropna=False), end="\n\n")

    # Création du tableau de contingence
    contingency_table = pd.crosstab(df[col1], df[col2], normalize=normalize)
    print(f"Tableau de contingence entre {col1} et {col2}:\n")
    print(contingency_table, end="\n\n")

    # Affichage du graphique
    plt.figure(figsize=(8, 6))
    sns.countplot(x=col1, data=df)
    plt.title(f"Comptage {col1}")
    plt.xlabel(col1)
    plt.show()


def display_describe(df, col1):
    """
    Affiche une description de la variable et un plot de type histogramme pour sa distribution

    :param df: DataFrame contenant les données
    :param col1: Nom de la première colonne
    """

    print("Describe de la variable: ", end="\n\n")
    print(df[col1].describe(), end="\n\n")

    # Création de l'histogramme
    plt.figure(figsize=(8, 6))
    # sns.histplot(df[col1], kde=True, bins=30)
    sns.boxplot(x=col1, data=df)

    plt.title(f"Distribution {col1}")
    plt.xlabel(col1)
    plt.show()


# Chargement des données
df = pd.read_csv("../data/save/ma_base_train_clean.csv")

# Affichage des informations du DataFrame
display_dataframe_info(df, title="Informations sur le DataFrame")

# matrice de confusion pour Policy_Sales_Channel
display_contingency_table(
    df, "Policy_Sales_Channel", "Response", normalize="index"
)

# On choisit d'utiliser les informations obtenues par le tableau de contingence pour
# réencoder la variable

display(
    df["Policy_Sales_Channel"]
    .value_counts(normalize=True, dropna=False)
    .head(5)
)

crosstab_policy = pd.crosstab(
    df["Policy_Sales_Channel"], df["Response"], normalize="index"
)

print("type de crosstab_policy : ", type(crosstab_policy))

df["Policy_Sales_Channel"] = np.where(
    df["Policy_Sales_Channel"].isin(
        crosstab_policy[crosstab_policy[1] >= 0.12].index.tolist()
    ),
    1,
    0,
)

# Test de Chi-Deux

# On vérifie la significativité de la variable
# On utilise le test de Chi-Deux pour vérifier l'association entre la variable
# Policy_Sales_Channel et la variable cible Response
stat, p = chi2_contingency(
    pd.crosstab(df["Response"], df["Policy_Sales_Channel"])
)[:2]
# On calcule le V de Cramer
# Le V de Cramer est une mesure de l'association entre deux variables catégorielles
V_Cramer = np.sqrt(
    stat / pd.crosstab(df["Response"], df["Policy_Sales_Channel"]).values.sum()
)

# On affiche la liste des canaux de communication qui prennent la valeur 1
print()
print(
    "La liste des codes des canaux de communication : \n\n",
    crosstab_policy[crosstab_policy[1] >= 0.12].index.tolist(),
    end="\n\n",
)

# On affiche la valeur du V de Cramer
print("Le V de Cramer est égal à : ", V_Cramer, end="\n\n")

# On affiche la p-valeur (pour trancher sur la significativité de la variable)
print("La p-valeur du test de Chi-Deux est égal à : ", p)

# matrice de confusion pour Policy_Sales_Channel
display_contingency_table(df, "Region_Code", "Response", normalize="index")

display_describe(df, "Region_Code")

IQR = df["Region_Code"].quantile(0.75) - df["Age"].quantile(0.25)
I1 = df["Region_Code"].quantile(0.25) - 1.5 * IQR
I2 = df["Region_Code"].quantile(0.75) + 1.5 * IQR
print("Region_Code :", IQR, end="\n\n")
print("InterRegion_Codevalle :[", I1, ";", I2, "]")

display(df.loc[(0 > df["Region_Code"]) | (df["Region_Code"] > 50)].head(10))

for i in pd.get_dummies(df["Region_Code"]):
    # Test du Chi-Deux

    stat, p = chi2_contingency(
        pd.crosstab(df["Response"], pd.get_dummies(df["Region_Code"])[i])
    )[:2]

    # V de Cramer
    V_Cramer = np.sqrt(
        stat / pd.crosstab(df["Response"], df["Region_Code"]).values.sum()
    )

    # On affiche uniquement les variables significatives et dont le V de Cramer est supérieur à 0.1

    if (p < 0.05) & (V_Cramer > 0.1):
        print(i, V_Cramer)

# Variable Region_Code

df["Region_Code"].value_counts(normalize=True, dropna=False).head(5)

crosstab_region = pd.crosstab(
    df["Region_Code"], df["Response"], normalize="index"
)

region_correlation = np.where(
    df["Region_Code"].isin(
        crosstab_region[crosstab_region[1] >= 0.12].index.tolist()
    ),
    1,
    0,
)

# Test du Chi-Deux
stat, p = chi2_contingency(pd.crosstab(df["Response"], region_correlation))[:2]

# V de Cramer

V_Cramer = np.sqrt(
    stat / pd.crosstab(df["Response"], region_correlation).values.sum()
)

print()

print("Le V de cramer est égal à : ", V_Cramer, end="\n\n")

print("La p-valeur du test de Chi-Deux est égal à : ", p)

# Prends la serie binaire (0,1) de la valeur de la variable Region_Code == 28
# et la transforme en variable binaire
df["Region_Code"] = pd.get_dummies(df["Region_Code"])[28]

# Tratement de la variable Vehicle_Date
# separation de la date en année, mois et jour de la semaine
# train["Year"] = train["Vehicle_Date"].dt.year
# train["Month"] = train["Vehicle_Date"].dt.month
# train["Day"] = train["Vehicle_Date"].dt.dayofweek + 1

# matrice de confusion pour Policy_Sales_Channel
# display_contingency_table(df, 'Annual_Premium', 'Response', normalize='index')

# Solution

# On transforme les valeurs en euros

df["Annual_Premium"] = df["Annual_Premium"] / 100

# On affiche le boxplot

sns.boxplot(x="Annual_Premium", data=df)

plt.title("Boxplot de la variable Annual_Premium ")

plt.show()

# On affiche l'histogramme

sns.histplot(df["Annual_Premium"], stat="density")

plt.title("Histogramme de la variable Annual_Premium ")

plt.show()

# Solution

# Histogramme de la variable Age

sns.histplot(df["Age"], stat="density")

plt.title("Histogramme de la variable Age ")

plt.show()

# Histogramme de la variable Vintage

sns.histplot(df["Vintage"], stat="density")

plt.title("Histogramme de la variable Vintage ")

plt.show()

# Solution

# Normalisation entre 0 et 1, la distribution n'est pas normale
# mais elle ne comporte pas d'outliers susceptible d'influer sur la normalisation
# On utilise le MinMaxScaler pour les variables Age et Vintage
# car elles sont continues et leur distribution est relativement uniforme
scaler1 = MinMaxScaler()

# Robust Scaling (présence d'outliers)
# On utilise le RobustScaler pour la variable Annual_Premium
# car elle peut contenir des valeurs extrêmes
# et sa distribution n'est pas normale
scaler2 = RobustScaler()

# On normalise entre 0 et 1 les variables Age et Vintage
df[["Age", "Vintage"]] = scaler1.fit_transform(df[["Age", "Vintage"]])

# On met à l'échelle la variable Annual_Premium (Robust Scaling)

df["Annual_Premium"] = scaler2.fit_transform(df[["Annual_Premium"]])

# Instanciation de OneHotEncoder
encoder = OneHotEncoder(
    sparse_output=False, drop=None
)  # sparse=False pour obtenir un array dense

# Fit et transformation sur la colonne à encoder
encoded_array = encoder.fit_transform(df[["Gender"]])

# Récupération des noms des colonnes créées (ex: Sexe_Female, Sexe_Male)
encoded_cols = encoder.get_feature_names_out(["Gender"])

# Création d'un DataFrame pandas à partir du tableau numpy encodé
encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)

# Concaténation du DataFrame d'encodage au DataFrame original
df = pd.concat([df, encoded_df], axis=1)

# Suppression de la colonne initiale
df = df.drop("Gender", axis=1)

# Fit et transformation sur la colonne à encoder
encoded_array = encoder.fit_transform(df[["Vehicle_Age"]])

# Récupération des noms des colonnes créées (ex: Sexe_Female, Sexe_Male)
encoded_cols = encoder.get_feature_names_out(["Vehicle_Age"])

# Création d'un DataFrame pandas à partir du tableau numpy encodé
encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)

# Concaténation du DataFrame d'encodage au DataFrame original
df = pd.concat([df, encoded_df], axis=1)

# Suppression de la colonne initiale
df = df.drop("Vehicle_Age", axis=1)

df = df.rename(
    columns={
        "Vehicle_Age_1-2 Year": "Vehicle_Damage_Middle",
        "Vehicle_Age_< 1 Year": "Vehicle_Damage_First",
        "Vehicle_Age_> 2 Years": "Vehicle_Damage_Last",
    }
)

# Fit et transformation sur la colonne à encoder
encoded_array = encoder.fit_transform(df[["Vehicle_Damage"]])

# Récupération des noms des colonnes créées (ex: Sexe_Female, Sexe_Male)
encoded_cols = encoder.get_feature_names_out(["Vehicle_Damage"])

# Création d'un DataFrame pandas à partir du tableau numpy encodé
encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)

# Concaténation du DataFrame d'encodage au DataFrame original
df = pd.concat([df, encoded_df], axis=1)

# Suppression de la colonne initiale
df = df.drop("Vehicle_Damage", axis=1)

# Fit et transformation sur la colonne à encoder
encoded_array = encoder.fit_transform(df[["Previously_Insured"]])

# Récupération des noms des colonnes créées (ex: Sexe_Female, Sexe_Male)
encoded_cols = encoder.get_feature_names_out(["Previously_Insured"])

# Création d'un DataFrame pandas à partir du tableau numpy encodé
encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)

# Concaténation du DataFrame d'encodage au DataFrame original
df = pd.concat([df, encoded_df], axis=1)

# Suppression de la colonne initiale
df = df.drop("Previously_Insured", axis=1)

df.to_csv("../data/save/ma_base_train_normalisée.csv", index=False)
