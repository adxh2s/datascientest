# Import des librairies
import pandas as pd
import numpy as np

# Transformation

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# Modèle

import statsmodels.formula.api as smf

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

# Rééchantillonnage 

from imblearn.over_sampling import SMOTE, RandomOverSampler

from imblearn.under_sampling import RandomUnderSampler

# Evaluation et métriques

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
    
    

# Chargement des données
df = pd.read_csv('../data/save/ma_base_train_normalisée.csv')

# Affichage des informations du DataFrame
display_dataframe_info(df, title="Informations sur le DataFrame")

logreg = LogisticRegression().fit(
    df.drop("Response", axis=1), df["Response"])

coef = pd.DataFrame(logreg.coef_, columns=df.drop(
    "Response", axis=1).columns).T.rename(columns={0: "Coefficients"})

coef.index.name = 'Variables'

print()

print(coef.to_markdown())

# Exemple avec 'y' comme variable cible
target = 'Response'

# Liste des colonnes explicatives : toutes sauf la cible
predictors = df.columns.drop(target)

# Construction de la chaîne de caractères de la formule
formula = target + ' ~ ' + ' + '.join(predictors)

# Modèle avec formule automatique
model = smf.ols(formula=formula, data=df)
results = model.fit()

print(results.summary())


# Liste des colonnes explicatives : toutes sauf la cible
predictors = df.columns.drop(target)

# Construction de la chaîne de caractères de la formule
formula = target + ' ~ ' + ' + '.join(predictors)
# Modèle de régression logistique avec statsmodels
# On utilise smf.logit pour la régression logistique
logit = smf.logit(formula, data=df).fit()
# Affichage du résumé du modèle
print("\nRésumé du modèle de régression logistique :")
print(logit.summary2())


# Récupération des coefficients
logit_ = logit.params
print("\nlogit_ :")
display(logit_)
# Rectification du coefficient de la variable Age 
logit_["Age"] = logit_["Age"] / (85 - 20)

# Rectification du coefficient de la variable Annual Premium
logit_["Annual_Premium"] = logit_["Annual_Premium"] / (39411 - 24421)

# Passage à l'exponentielle
logit_ = np.exp(logit_)

# Variables ayant un coefficient négatif
logit_[["Gender", "Age", "Previously_Insured"]] = - \
    1 / logit_[["Gender", "Age", "Previously_Insured"]]

# Mise en forme 
print()
print(pd.DataFrame(logit_, columns=[
      "Odd Ratios"], index=logit_.index).to_markdown())