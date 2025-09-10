import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
    # Insérez votre code ici
import statsmodels.api as sm    
from statsmodels.tsa.seasonal import seasonal_decompose
from IPython.display import display


# print(f"Working directory: {os.getcwd()}")
# print(f"Script location: {__file__}")
# print(f"Files in current dir: {os.listdir('.')}")
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


# chargement des données
print("🔄 Chargement des données...")
try:
    airpass = pd.read_csv('data/AirPassengers.csv', header=0, index_col=0, parse_dates=True)
    display(airpass.head())
    print("✅ Données chargées avec succès!")
    print()

    # application du log sur les données (passage en échelle log additive)
    airpasslog = np.log(airpass)
    display(airpasslog.head())

    # Affichage riche du DataFrame
    # display_dataframe_info(df, "AirPassengers Data")
    # Exemple d'exploration d'une colonne (à décommenter si nécessaire)
    plt.plot(airpass)
    plt.title('Air Passengers Over Time')
    plt.show()

    # Transformation logarithmique pour stabiliser la variance
    plt.plot(airpasslog)
    plt.title('Log Transformed Data')
    plt.show()

    # Stationarité de la série
    # exemple
    epsilon = np.random.rand(100) #Bruit Blanc
    t = np.linspace(0,10,100) # Temps
    alpha = 1.5 #coefficient de tendance
    total = pd.DataFrame(alpha * t + epsilon)
    plt.plot(total)
    plt.show()

    # diffœrenciation
    total_diff = total.diff().dropna()
    plt.plot(total_diff)
    plt.show()
    
    # On revient sur la série airpasslog
    # Autocorrélation
    pd.plotting.autocorrelation_plot(airpasslog)
    plt.title('Autocorrelation Plot')
    plt.show()
    
    # Création de la figure et des axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7)) 
     # Différenciation ordre 1
    airpasslog_1 = airpasslog.diff().dropna()
    # Série temporelle différenciée
    airpasslog_1.plot(ax = ax1) 
    ax1.set_title('Série différenciée d\'ordre 1')
    # Autocorrélogramme de la série différenciée
    pd.plotting.autocorrelation_plot(airpasslog_1, ax = ax2)
    ax2.set_title('Autocorrélogramme de la série différenciée d\'ordre 1')
    plt.show()

    # Création de la figure et des axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7)) 
     # 2eme Différenciation d'ordre 12
    airpasslog_2 = airpasslog_1.diff(periods=12).dropna()
    # Série temporelle doublement différenciée
    airpasslog_2.plot(ax = ax1) 
    ax1.set_title('Série doublement différenciée')
    # Autocorrélogramme de la série différenciée
    pd.plotting.autocorrelation_plot(airpasslog_2, ax = ax2)
    ax2.set_title('Autocorrélogramme de la série doublement différenciée')
    plt.show()

    # Test de Dickey-Fuller augmenté
    _, p_value, _, _, _, _  = sm.tsa.stattools.adfuller(airpasslog_2)
    print(p_value)  # p-valeur bien inférieure à 0.05, on peut considérer la série comme stationnaire.

    input("Press Enter to continue...")
except FileNotFoundError:
    print("❌ Erreur: Le fichier 'data/AirPassengers.csv' n'a pas été trouvé.")
    print("Vérifiez que le fichier existe dans le dossier 'data'.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    sys.exit(1)
