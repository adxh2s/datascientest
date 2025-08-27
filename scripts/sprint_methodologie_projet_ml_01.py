from sklearn.model_selection import train_test_split

# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
# Matplotlib pour la visualisation
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch

# kds pour les métriques et visualisations
import kds

# Seaborn pour des visualisations plus esthétiques
import seaborn as sns

# Ydata Profiling pour le rapport de données
# Assurez-vous d'avoir installé ydata-profiling avec `pip install ydata-profiling`
# Si vous utilisez une version antérieure à 4.0, utilisez `from pandas_profiling import ProfileReport`
# Pour la version 4.0 et ultérieure, utilisez :
from ydata_profiling import ProfileReport

# Pour la génération de nombres aléatoires
import random

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
        
        
def plot_confusion_matrix_seaborn(cm, figsize=(6,5), variable_name=None):
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
    row_labels = ['Vrai Négatif', 'Faux Négatif']
    col_labels = ['Faux Positif', 'Vrai Positif']

    # Convertir en DataFrame pour seaborn (avec labels)
    df_cm = pd.DataFrame(cm, index=row_labels, columns=col_labels)
    
    # Couleurs personnalisées pour chaque catégorie
    # On crée une palette map par catégories pour colorer le fond via un DataFrame semblable
    # Vert clair pour VP/VN, rouge clair pour FP/FN
    colors_array = np.array([['#97DBAE', '#FF6978'],
                             ['#FF6978', '#97DBAE']])
    df_colors = pd.DataFrame(colors_array, index=row_labels, columns=col_labels)

    # Création de la figure
    plt.figure(figsize=figsize)

    # Heatmap avec cmap neutre et annot
    ax = sns.heatmap(df_cm, annot=True, fmt='d', cbar=False,
                     linewidths=0.8, linecolor='black',
                     square=True, cmap='Greys',
                     mask=None,
                     annot_kws={"weight": "bold", "size": 14},
                     )

    # Colore le fond des cases individuellement avec les couleurs définies (semi-transparent)
    for y in range(df_cm.shape[0]):
        for x in range(df_cm.shape[1]):
            ax.add_patch(
                plt.Rectangle((x, y), 1, 1, fill=True, facecolor=df_colors.iat[y, x], alpha=0.3, edgecolor='none', lw=0)
            )

    # Construction du titre avec variable_name
    if variable_name:
        plt.title(f"Matrice de confusion pour la variable : {variable_name}", fontsize=16)
    else:
        plt.title("Matrice de confusion", fontsize=16)

    # Légende
    legend_elements = [
        Patch(facecolor='#97DBAE', edgecolor='black', label='Vrais Positifs / Vrais Négatifs'),
        Patch(facecolor='#FF6978', edgecolor='black', label='Faux Positifs / Faux Négatifs')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1), title='Légende')

    # Labels axes
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Vérité terrain')

    plt.tight_layout()
    plt.show()

    
# Chargement des données
df = pd.read_csv('../data/base.csv')

display_dataframe_info(df, "Insurance Prédiction DataFrame")

# Séparation des variables explicatives et de la cible
X = df.drop('Response', axis=1)
y = df['Response']

# Séparation des données en ensembles d'entraînement et de test
# En utilisant stratification pour conserver la distribution de la cible
X_train, y_train, X_test, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    stratify=y,
                                                    random_state=42)

print('Shape X train, X test, y train, y test : ', X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print('Value_count: ', y.value_counts(normalize=True))

# Base test
test = df.sample(frac=0.1, random_state=42)

# Export de la base de test
test.to_csv("../data/save/test_raw_copy.csv", index=False)

# Base d'entraînement
train = df.drop(test.index)

# Export de la base d'entraînement
train.to_csv("../data/save/train_raw_copy.csv", index=False)

display(train["Response"].value_counts())

display(train[["Gender", "Driving_License", "Region_Code", "Previously_Insured",
       "Vehicle_Damage", "Policy_Sales_Channel"]].nunique())

display(train.dtypes)

display(df.info())

display(train.isna().sum())

# Valeurs manquantes

print()
print("Proportion des valeurs manquantes: ", end="\n\n")
print(train["Gender"].value_counts(dropna=False, normalize=True), end="\n\n")

# Variables continues

print("La valeur médiane selon le sexe du client: ", end="\n\n")
print(train[['Age', 'Annual_Premium', 'Vintage',
             'Gender']].groupby("Gender").median(),
      end="\n\n")

# Variables catégorielles

print("Le mode selon le sexe du client: ", end="\n\n")
print(train[[
    'Gender', 'Region_Code', 'Previously_Insured', 'Vehicle_Damage',
    'Policy_Sales_Channel'
]].groupby("Gender").apply(pd.DataFrame.mode).set_index("Gender"),
      end="\n\n")

# Vérification des proportions

print("Proportion: ", end="\n\n")
print(
    train.groupby("Previously_Insured")['Gender'].value_counts(normalize=True),
    end="\n\n")

print(train.groupby("Vehicle_Damage")['Gender'].value_counts(normalize=True),
      end="\n\n")

# Nous choisissons d'imputer la variable à l'aide des informations récoltées précédemment

print()

print("Evaluation de la part des données manquantes: ", end="\n\n")

print(train['Gender'].value_counts(dropna=False, normalize=True), end="\n\n")

# 1) On impute à l'aide des valeurs observées dans les autres variables

train['Gender'] = np.where(
    (train['Gender'].isna()) &
    (((abs(train['Age'] - 30)) >
      (abs(train['Age'] - 41))) & (train['Previously_Insured'] == 0) &
     (train['Vehicle_Damage'] == 1)), 0, train['Gender'])

print("Evaluation de la part des données manquantes restante: ", end="\n\n")

print(train['Gender'].value_counts(dropna=False, normalize=True), end="\n\n")

# 2) On impute le reste des données manquantes aléatoirement de telle sorte à conserver la proportion initiale

proportion = [0] * 55 + [1] * 45

train.loc[train['Gender'].isna(),
          'Gender'] = train.loc[train['Gender'].isna(), 'Gender'].apply(
              lambda x: random.choice(proportion))

# Vérification

print("Vérification: ", end="\n\n")
print(train['Gender'].value_counts(dropna=False, normalize=True), end="\n\n")

# Données manquantes

print()

pourcentage = round(train['Policy_Sales_Channel'].isna().sum() / len(train),
                    2) * 100

print(f"Il y a {pourcentage} % de données manquantes.", end="\n\n")

# Canaux majoritaires

nombre = train['Policy_Sales_Channel'].nunique()

print(f'Il y a {nombre} canaux.', end="\n\n")

print(train['Policy_Sales_Channel'].value_counts(normalize=True,
                                                 dropna=False).head(10),
      end="\n\n")

# Les canaux 152, 26 et 124 sont majoritaires.

# SIndexation sur la colonne 'id' pour faciliter les opérations futures
train.set_index('id', inplace=True)

# Variable Driving_License

print()

# Répartition des catégories

print("Répartion des catégories: ", end="\n\n")

print(train['Driving_License'].value_counts(normalize=True), end="\n\n")


def display_contingency_table(df, col1, col2, normalize):
    """
    Affiche un tableau de contingence entre deux colonnes d'un DataFrame.
    
    :param df: DataFrame contenant les données
    :param col1: Nom de la première colonne
    :param col2: Nom de la deuxième colonne
    """
    
    print("Répartition de la variable: ", end="\n\n")
    print(df[col1].value_counts(normalize=True, dropna=False),
          end="\n\n")
    print(df[col2].value_counts(normalize=True, dropna=False),
          end="\n\n")    
    
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
    
    
# Affichage du tableau de contingence entre Driving_License et Response
display_contingency_table(train, 'Driving_License', 'Response', normalize=True)

# Variable Region_Code
# Code Région le plus faible
print()
print('Code région le plus faible:', train['Region_Code'].min(), end="\n\n")

# Code Région le plus élevé
print('Code région le plus élevé:', train['Region_Code'].max(), end="\n\n")

# Les 5 régions les plus représentées
print("Les 5 régions les plus représentées sont : ", end="\n\n")
print(train['Region_Code'].value_counts(normalize=True).head(), end="\n\n")

# Affichage du tableau de contingence entre Region_Code et Response
display_contingency_table(train, 'Region_Code', 'Response', normalize=True)

# Choisir une palette de couleurs proposée par Seaborn
colors = sns.color_palette("pastel")[0:5]
count = train['Region_Code'].value_counts()
labels = count.index
# Création du camembert
plt.pie(
    count,
    labels=labels,
    colors=colors,
    autopct="%.0f%%",        # Affiche le % dans chaque part
    startangle=90            # Pour démarrer à 90°
)

plt.title("Exemple de camembert avec Seaborn + Matplotlib")
plt.show()

# Affichage du tableau de contingence entre Region_Code et Response
display_contingency_table(train, 'Previously_Insured', 'Response', normalize='index')

# Affichage du tableau de contingence entre Region_Code et Response
display_contingency_table(train, 'Vehicle_Damage', 'Response', normalize='index')

display_describe(train, 'Vintage')

display_describe(train, 'Annual_Premium')


display_describe(train, 'Age')

IQR = train["Age"].quantile(0.75)-train["Age"].quantile(0.25)
I1 = train["Age"].quantile(0.25) - 1.5 * IQR
I2 = train["Age"].quantile(0.75) + 1.5 * IQR
print("IQR :", IQR, end="\n\n")
print("Intervalle :[", I1, ";", I2, "]")

display(df.loc[(18 > df['Age']) | (df['Age'] > 85)].head(10))

train.to_csv("../data/save/ma_base_train_clean.csv", index=False)

fig, ax = plt.subplots(3, 3, figsize=(20, 15), dpi=200)

fig.tight_layout(pad=5)

ax[0, 0].pie(train['Gender'].value_counts(normalize=True),
             labels=['Hommes', 'Femmes'],
             explode=[0.05, 0],
             autopct="%0.2f%%",
             colors=["silver", "wheat"])

ax[0, 0].text(-1.2,
              1.75,
              'Répartition hommes/femmes',
              fontsize=15,
              weight='bold')

sns.boxplot(ax=ax[0, 1], x="Vintage", data=train, palette="light:b")

ax[0, 1].spines['right'].set_visible(False)

ax[0, 1].spines['top'].set_visible(False)

ax[0, 1].yaxis.set_ticks_position('left')

ax[0, 1].xaxis.set_ticks_position('bottom')

ax[0, 1].text(30,
              -0.7,
              "Distribution de la variable vintage",
              fontsize=15,
              weight='bold')

sns.barplot(ax=ax[0, 2],
            x=['Non', 'Oui'],
            y=train["Previously_Insured"].value_counts(normalize=True),
            palette=["thistle", "tan"])

ax[0, 2].spines['right'].set_visible(False)

ax[0, 2].text(-0.3,
              0.68,
              "Proportion de clients déjà assurés",
              fontsize=15,
              weight='bold')

sns.barplot(
    ax=ax[1, 0],
    x=['152', '26', '124', '160', '156'],
    y=train["Policy_Sales_Channel"].value_counts(normalize=True).head(5),
    palette="magma")

ax[1, 0].spines['right'].set_visible(False)

ax[1, 0].spines['top'].set_visible(False)

ax[1, 0].yaxis.set_ticks_position('left')

ax[1, 0].xaxis.set_ticks_position('bottom')

ax[1, 0].text(-0.2,
              0.4,
              "Top 5 Canaux de Communication",
              fontsize=15,
              weight='bold')

ax[1, 1].pie(train["Response"].value_counts(normalize=True),
             labels=['Pas intéressé', 'Intéressé'],
             explode=[0.05, 0.05],
             autopct="%0.2f%%",
             colors=["palevioletred", "darksalmon"])

ax[1, 1].text(-1,
              1.4,
              "Répartition variable cible",
              fontsize=15,
              weight='bold')

sns.boxplot(ax=ax[1, 2], x="Annual_Premium", data=train, color="rosybrown")

ax[1, 2].spines['right'].set_visible(False)

ax[1, 2].spines['top'].set_visible(False)

ax[1, 2].yaxis.set_ticks_position('left')

ax[1, 2].xaxis.set_ticks_position('bottom')

ax[1, 2].text(-0.1,
              -0.55,
              "Distribution de la prime d'assurance",
              fontsize=15,
              weight='bold')

sns.histplot(ax=ax[2, 0],
             x="Age",
             data=train,
             color="brown",
             stat="density",
             kde=True)

ax[2, 0].spines['right'].set_visible(False)

ax[2, 0].spines['top'].set_visible(False)

ax[2, 0].yaxis.set_ticks_position('left')

ax[2, 0].xaxis.set_ticks_position('bottom')

ax[2, 0].text(35, 0.112, "Distribution de l'âge", fontsize=15, weight='bold')

sns.barplot(ax=ax[2, 1],
            x=['28', '8', '46', '41', '15'],
            y=train["Region_Code"].value_counts(normalize=True).head(5),
            palette="dark:salmon_r")

ax[2, 1].spines['right'].set_visible(False)

ax[2, 1].spines['top'].set_visible(False)

ax[2, 1].yaxis.set_ticks_position('left')

ax[2, 1].xaxis.set_ticks_position('bottom')

ax[2, 1].text(0.9, 0.32, "Top 5 Code Région", fontsize=15, weight='bold')

ax[2, 2].pie(train['Vehicle_Damage'].value_counts(normalize=True),
             labels=['Oui', 'Non'],
             explode=[0.05, 0],
             autopct="%0.2f%%",
             colors=["indianred", "sienna"])

ax[2, 2].spines['right'].set_visible(False)

ax[2, 2].spines['top'].set_visible(False)

ax[2, 2].yaxis.set_ticks_position('left')

ax[2, 2].xaxis.set_ticks_position('bottom')

ax[2, 2].text(-1.1, 1.45, "Dommages sur véhicule", fontsize=15, weight='bold')

plt.show()

fig.savefig('full_figure.png', dpi=600)


profile = ProfileReport(df, title="Rapport profilage de données", explorative=True)
profile.to_file("../data/html/rapport_ydata.html")