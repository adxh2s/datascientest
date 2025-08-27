import pandas as pd
import numpy as np

def detect_dtype_from_values(series):
    """Détecte un type généralisé basé sur le contenu réel de la série."""
    s_non_na = series.dropna()
    if s_non_na.empty:
        return "inconnu"

    dtype = series.dtype

    # Détection bas niveau pour dtype réel
    if pd.api.types.is_bool_dtype(series):
        return 'bool'
    elif pd.api.types.is_integer_dtype(series):
        return 'int'
    elif pd.api.types.is_float_dtype(series):
        return 'float'
    elif pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    
    # Tentative : datetime convertible
    try:
        parsed = pd.to_datetime(s_non_na, errors='coerce')
        if parsed.notna().all():
            return 'datetime (convertible)'
    except:
        pass

    # Tentative : numérique convertible
    try:
        parsed_num = pd.to_numeric(s_non_na.astype(str).str.replace(',', '.'), errors='coerce')
        if parsed_num.notna().all():
            return 'numeric (convertible)'
    except:
        pass

    # Tentative : bool convertible
    str_vals = s_non_na.astype(str).str.lower().unique()
    if set(str_vals) <= {'true', 'false', '0', '1', 'yes', 'no', 'vrai', 'faux'}:
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


def test_conversion_possibility(series):
    col_non_na = series.dropna()

    # Date
    try:
        if pd.to_datetime(col_non_na, errors='coerce').notna().all():
            return 'to_datetime'
    except:
        pass

    # Numeric
    try:
        if pd.to_numeric(col_non_na, errors='coerce').notna().all():
            return 'to_numeric'
    except:
        pass

    # Bool
    is_bool, count_bool = detect_boolean(series)
    if is_bool and count_bool >= len(col_non_na):
        return 'to_bool'

    return ''


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

        # Détection du type réel
        detected = detect_dtype_from_values(serie)
        ligne['type_détecté'] = detected
        ligne['conforme_dtype'] = is_conforme_dtype(serie, detected)
        ligne['dtype_suggéré'] = detected if not ligne['conforme_dtype'] else ''

        # Détection de textes, nombres, booléens, dates
        is_num, count_num = detect_numeric_in_object(serie)
        is_bool, count_bool = detect_boolean(serie)
        ligne['is_number'] = is_num
        ligne['count_number'] = count_num
        ligne['is_boolean'] = is_bool
        ligne['count_boolean'] = count_bool
        ligne['is_text'] = pd.api.types.is_string_dtype(serie)
        ligne['is_datetime'] = pd.api.types.is_datetime64_any_dtype(serie)

        # Proposition de conversion
        ligne['conversion_possible'] = test_conversion_possibility(serie)

        all_info.append(ligne)

    return pd.DataFrame(all_info).set_index('colonne')


# Configuration pour un affichage plus riche
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

def display_dataframe_info(df, title="DataFrame Info"):
    """Affiche les informations du DataFrame de manière formatée"""
    print("=" * 80)
    print(f"📊 {title}")
    print("=" * 80)
    
    # Informations de base
    print(f"📋 Forme du DataFrame: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"💾 Utilisation mémoire: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
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
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print("🏷️  Aperçu des valeurs uniques (colonnes catégorielles):")
        print("-" * 40)
        for col in categorical_cols[:5]:  # Limite à 5 colonnes pour éviter l'encombrement
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
    
    if df[column_name].dtype == 'object':
        print("\nValeurs les plus fréquentes:")
        print(df[column_name].value_counts().head(10))
    elif df[column_name].dtype in ['int64', 'float64']:
        print(f"\nMin: {df[column_name].min()}")
        print(f"Max: {df[column_name].max()}")
        print(f"Moyenne: {df[column_name].mean():.2f}")
        print(f"Médiane: {df[column_name].median():.2f}")