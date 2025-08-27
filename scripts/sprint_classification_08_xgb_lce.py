import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_validate, KFold, GridSearchCV
from sklearn.metrics import f1_score
# Pandas et Numpy pour dataframe et calcul
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
df = pd.read_csv('../data/adult_income.csv')

display_dataframe_info(df, "Revenus Adultes DataFrame")

# Remplacer les '?' par NaN
df = df.replace('?', np.nan)
df = df.drop('education', axis=1)

df.replace(['Cambodia', 'China', 'Hong', 'India', 'Iran', 'Japan',
               'Laos', 'Philippines', 'Taiwan', 'Thailand', 'Vietnam'], 'Asia', inplace=True)

df.replace(['Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', 'El-Salvador',
               'Guatemala', 'Haiti', 'Honduras', 'Jamaica', 'Mexico', 'Nicaragua',
                'Peru', 'Puerto-Rico', 'Trinadad&Tobago', 'South'], 'Center & South America', inplace=True)

df.replace(['England', 'France', 'Germany', 'Greece', 'Holand-Netherlands', 'Hungary', 
               'Ireland', 'Italy', 'Poland', 'Portugal', 'Scotland', 'Yugoslavia'], 'Europe', inplace=True)

df.replace(['United-States', 'Canada'], 'Canada&USA', inplace=True)
display_dataframe_info(df, "Revenus Adultes DataFrame (After Cleaning)")

# explore_column(df, 'Outcome')
features = df.drop('income', axis=1)
target = df['income']

# Dichotomisation de la variable cible
target =  [1 if x==">50K" else 0 for x in target]

# dichotomisation des variables catégorielles
features_matrix = pd.get_dummies(features)
print(f"Nombre de variables après dichotomisation: {features_matrix.shape[1]}")
# Affichage des premières lignes de la matrice de caractéristiques
display(features_matrix.head())
# Séparation des données en 3 ensembles : entraînement, test et validation
# Données de validation
X, X_valid, y, y_valid = train_test_split(features_matrix, target, test_size=0.1)
# Décomposition des données restantes en deux ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Conversion des données en DMatrix pour XGBoost
# DMatrix est une structure de données optimisée pour XGBoost
train = xgb.DMatrix(data=X_train, label=y_train)
test = xgb.DMatrix(data=X_test, label=y_test)
valid = xgb.DMatrix(data=X_valid, label=y_valid)

# Paramétrage du modèle XGBoost 1
params = {'booster': 'gbtree', 'learning_rate': 1, 'objective': 'binary:logistic'}
xgb1 = xgb.train(params=params, dtrain=train, num_boost_round=100, evals=[(train, 'train'), (test, 'eval')])

# Paramétrage du modèle XGBoost 2
params = {'booster': 'gbtree', 'learning_rate': 0.01, 'objective': 'binary:logistic'}
xgb2 = xgb.train(params=params, dtrain=train, num_boost_round=700, evals=[(train, 'train'), (test, 'eval')])

# Graphique de l'importance des caractéristiques
# xgb.plot_importance(xgb2, max_num_features=15)
# plt.show()

# types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
# for f in types:
#     xgb.plot_importance(xgb2, max_num_features=15, importance_type=f, title='importance: '+f)

# plt.show()

# Cross-validation pour évaluer le modèle
bst_cv = xgb.cv(params=params,
                dtrain=train,
                num_boost_round=100,
                nfold=3,
                early_stopping_rounds=60)
display(bst_cv)

# Prédictions sur l'ensemble de test
preds = xgb2.predict(test)
# Conversion des prédictions en classes binaires
# On considère un seuil de 0.5 pour la classification binaire
xgbpreds = pd.Series(np.where(preds > 0.5, 1, 0))
# Affichage des prédictions
print("🔍 Matrice de confusion pour le XGBooster:")
display(pd.crosstab(xgbpreds, pd.Series(y_test), rownames=['Classe réelle'], colnames=['Classe prédite']))

# Affichage de l'erreur de prédiction sur valid
print("🔍 Erreur de prédiction sur l'ensemble de validation:")
print(xgb2.eval(valid))

# Classification avec LCE
from lce import LCEClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lce_clf = LCEClassifier(n_estimators=2, n_jobs=-1, random_state=0)
lce_clf.fit(X_train, y_train)

y_pred = lce_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.1f}%".format(accuracy*100))

cm = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
display(cm)