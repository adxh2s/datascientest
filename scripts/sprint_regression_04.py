import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.linear_model import (
    LinearRegression,
    ElasticNetCV
)
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

def detect_dtype_from_values(series):
    """Détecte le type dominant des valeurs."""
    s_non_na = series.dropna()
    if s_non_na.empty:
        return "inconnu"

    if pd.api.types.is_bool_dtype(series):
        return 'bool'
    if pd.api.types.is_numeric_dtype(series):
        return 'float' if pd.api.types.is_float_dtype(series) else 'int'
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    
    # Tentative de parsing en datetime
    try:
        parsed = pd.to_datetime(s_non_na, errors='coerce')
        if parsed.notna().all():
            return 'datetime (convertible)'
    except:
        pass
    
    # Tentative de parsing numérique
    try:
        parsed_num = pd.to_numeric(s_non_na, errors='coerce')
        if parsed_num.notna().all():
            return 'numeric (convertible)'
    except:
        pass

    # Vérif bool
    bool_vals = set(s_non_na.astype(str).str.lower().unique())
    if bool_vals <= {'true', 'false', '0', '1'}:
        return 'bool (convertible)'

    return 'object'

def is_conforme_dtype(series, detected_type):
    declared = str(series.dtype)
    return (
        detected_type.startswith(declared) or 
        ("datetime" in detected_type and "datetime" in declared) or
        (declared == "object" and detected_type != "object")
    )

def detect_numeric_in_object(series):
    s = series.dropna().astype(str)
    parsed = s.apply(lambda v: v.replace(',', '.').strip())
    as_number = pd.to_numeric(parsed, errors='coerce')
    return as_number.notna().any(), as_number.notna().sum()

def detect_boolean(series):
    s = series.dropna().astype(str).str.lower()
    possible = {'true', 'false', '0', '1'}
    bool_detect = s.isin(possible)
    return bool_detect.any(), bool_detect.sum()


def describe_all(df):
    all_info = []

    for col in df.columns:
        serie = df[col]
        ligne = {
            'colonne': col,
            'dtype': str(serie.dtype),
            'non_nuls': serie.count(),
            'nuls': serie.isna().sum(),
            '%_nuls': round(serie.isna().mean()*100, 2),
            'nb_uniques': serie.nunique(dropna=True)
        }

        # Mode
        s_clean = serie.dropna()
        if not s_clean.empty:
            ligne['valeur_mode'] = s_clean.mode().iloc[0]
            ligne['freq_mode'] = s_clean.value_counts().iloc[0]
        else:
            ligne['valeur_mode'] = ligne['freq_mode'] = np.nan

        # Stats numériques
        if pd.api.types.is_numeric_dtype(serie):
            desc = serie.describe()
            ligne['moyenne'] = desc.get('mean', np.nan)
            ligne['ecart_type'] = desc.get('std', np.nan)
            ligne['min'] = desc.get('min', np.nan)
            ligne['25%'] = desc.get('25%', np.nan)
            ligne['médiane'] = desc.get('50%', np.nan)
            ligne['75%'] = desc.get('75%', np.nan)
            ligne['max'] = desc.get('max', np.nan)
        else:
            for stat in ['moyenne', 'ecart_type', 'min', '25%', 'médiane', '75%', 'max']:
                ligne[stat] = np.nan


        all_info.append(ligne)

    return pd.DataFrame(all_info).set_index('colonne')


# Charger le dataset
df = pd.read_csv("../data/2023_nba_player_stats.csv")

# Afficher les informations du DataFrame
display(describe_all(df))

# Mise à jour de l'index pour une meilleure visualisation
df = df.set_index(df["PName"] + " - " + df["Team"])
display(df.head())

# Suprimer les valeurs manquantes
df = df.dropna()
# Afficher les informations du DataFrame après nettoyage
display(df.isna().sum().sum())
display(df.POS.unique())
display(df.POS.value_counts())
display(df.POS.describe())
# suppression valeur 'G' de la colonne 'pos'
df = df[df['POS']!='G']

# Sélection des colonnes pertinentes
pl_pos = df['POS']
df = df.drop(columns=['PName', 'Team', 'POS'], axis=1)
display(df.head())
display(pl_pos)

# encodage de la colonne 'pos' en variables indicatrices
# methode 1: avec pd.get_dummies et join en une seule étape
df = df.join(pd.get_dummies(pl_pos, prefix='POS'))
# methode 2: en 2 étapes
# Etape 1 avec OneHotEncoder
# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# Avec tableau à 1D --> tableau à 2D (1 colonne, n lignes)
# encoded_df = pd.DataFrame(
#     encoder.fit_transform(pl_pos.values.reshape(-1, 1)).toarray(),
#     columns=encoder.get_feature_names_out([pl_pos.name]),
#     index=pl_pos.index
# )
# Etape 2 - 1 : avec la méthode join
# df = df.join(encoded_df)
# Etapde 2 - 2 avec pd.concat
# df = pd.concat([df, encoded_df], axis=1)

# Sélection des colonnes significatives
X = df.drop(['PTS'], axis=1)
y = df['PTS']

# Séparation des données d'entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# Standardisation des données
def scale_dataframe(scaler, df):
    """Standardise un DataFrame en utilisant un scaler."""
    return pd.DataFrame(
        scaler.transform(df),
        columns=df.columns,
        index=df.index
    )


# On entraine le modele
scaler = preprocessing.StandardScaler().fit(X_train)
#  On l'applique ensuite aux différents jeu de données disponibles
X_train_scaled = scale_dataframe(scaler, X_train)
X_test_scaled = scale_dataframe(scaler, X_test)

# matrice de corrélation
fig = plt.figure(figsize=(20, 20))
sns.heatmap(data=df.corr(), annot=True, cmap="RdBu_r", center=0)
plt.show()

# modele type regression linéaire sur une seule variable
lr1 = LinearRegression()
# entrainement du modele
lr1.fit(X_train_scaled[['Min']], y_train)

# Score R² 
print('Score (R²) Train :', lr1.score(X_train_scaled[['Min']], y_train))
print('Score (R²) Test :',  lr1.score(X_test_scaled[['Min']], y_test))

# prédiction
y_pred_train = lr1.predict(X_train_scaled[['Min']])
y_pred_test = lr1.predict(X_test_scaled[['Min']])
print('RMSE - train - LR variable Min: ', np.sqrt(mean_squared_error(y_train, y_pred_train)))
print('RMSE - test  - LR variable Min: ', np.sqrt(mean_squared_error(y_test, y_pred_test)))

# Modèle de regression linéaire pénalisée sur la totalité des variables
# Initialisation du modèle ElasticNet avec validation croisée
encv = ElasticNetCV(cv=8, l1_ratio=(0.1, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.99), alphas=(0.001, 0.01, 0.02, 0.025, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0))

# Entraînement du modèle
encv.fit(X_train_scaled, y_train)

coeffs = list(encv.coef_)
coeffs.insert(0, encv.intercept_)
feats = list(X.columns)
feats.insert(0, 'intercept')
display(pd.DataFrame({'valeur estimée': coeffs}, index=feats))

# Score R² 
print('Score (R²) Train EN Model: ', encv.score(X_train_scaled, y_train))
print('Score (R²) Test  EN Model: ',  encv.score(X_test_scaled, y_test))

# prédiction
y_pred_train = encv.predict(X_train_scaled)
y_pred_test = encv.predict(X_test_scaled)
print('RMSE - train - EN Model: ', np.sqrt(mean_squared_error(y_train, y_pred_train)))
print('RMSE - test  - EN Model: ', np.sqrt(mean_squared_error(y_test, y_pred_test)))

display(pd.DataFrame({'points_observés': y_test, 'points_predits': np.round(y_pred_test)}, index=X_test.index))