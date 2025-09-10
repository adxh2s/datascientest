import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
    # Insérez votre code ici
import statsmodels.api as sm    
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
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

    # application du log sur les données (passage en échelle log additive) Transformée logarithmique
    airpasslog = np.log(airpass)
    display(airpasslog.head())

    airpasslog_1 = airpasslog.diff().dropna() #Differenciation simple 
    airpasslog_2 = airpasslog_1.diff(periods = 12).dropna() #Différenciation d'ordre 12

    # Trace des fonctions d'autocorrélation et d'autocorrélation partielle
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7))
    # autocorrélation
    plot_acf(airpasslog_2, lags = 36, ax=ax1)
    ax1.set_title('Fonction d\'autocorrélation')
    # autocorrélation partielle
    ax2.set_title('Fonction d\'autocorrélation partielle')
    plot_pacf(airpasslog_2, lags = 36, ax=ax2)
    # Affichage
    plt.show()

    # Ajustement d'un modèle SARIMA
    # d = 1 (différenciation simple)
    # D = 1 (différenciation saisonnière)
    # p = 1 (ordre de l'AR)
    # q = 1 (ordre du MA)
    # P = 0 (ordre de l'AR saisonnier)
    # Q = 1 (ordre du MA saisonnier)
    # m = 12 (période saisonnière mensuelle)
    # (p,d,q)(P,D,Q)m avec p=1,d=1,q=1,P=0,D=1,Q=1,m=12
    model=sm.tsa.SARIMAX(airpasslog,order=(1,1,1),seasonal_order=(0,1,1,12))
    sarima=model.fit()
    print(sarima.summary())

    # On relance en ajoutant p=0 car la p-value était trop élevée avec le modèle précédent
    model = sm.tsa.SARIMAX(airpasslog,order=(0,1,1),seasonal_order=(0,1,1,12))
    sarima=model.fit()
    print(sarima.summary())

    import datetime
    pred = np.exp(sarima.predict(132, 143))#Prédiction et passage à l'exponentielle

    airpasspred = pd.concat([airpass, pred])#Concaténation des prédictions

    plt.plot(airpasspred) #Visualisation
    plt.axvline(x= datetime.date(1960,1,1), color='red'); # Ajout de la ligne verticale
    plt.title('Prédictions SARIMA')
    plt.show()  

    prediction = sarima.get_forecast(steps =12).summary_frame()  #Prédiction avec intervalle de confiance
    # Préparation du graphique
    fig, ax = plt.subplots(figsize = (15,5))
    # afficage de la série initiale
    plt.plot(airpass)
    prediction = np.exp(prediction) #Passage à l'exponentielle
    # Moyenne prédite
    prediction['mean'].plot(ax = ax, style = 'k--')
    # Intervalle de confiance
    ax.fill_between(prediction.index, prediction['mean_ci_lower'], prediction['mean_ci_upper'], color='k', alpha=0.1)
    plt.title('Prévisions SARIMA pour les 12 prochains mois')
    plt.show()

    input("Press Enter to continue...")
except FileNotFoundError:
    print("❌ Erreur: Le fichier 'data/AirPassengers.csv' n'a pas été trouvé.")
    print("Vérifiez que le fichier existe dans le dossier 'data'.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    sys.exit(1)
