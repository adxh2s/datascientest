import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
    # Insérez votre code ici
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from IPython.display import display


print(f"Working directory: {os.getcwd()}")
print(f"Script location: {__file__}")
print(f"Files in current dir: {os.listdir('.')}")
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
    df = pd.read_csv('data/AirPassengers.csv', header=0, index_col=0, parse_dates=True)
    display(df.head())
    print("✅ Données chargées avec succès!")
    print()
    
    # On créé une colonne décalée de 1 mois
    df['Passengers_lag1'] = df['#Passengers'].shift(1)
    display(df.head())
    # On créé une moyenne mobile sur 3 mois sur le nombre de passagers
    df['Passengers_MA_3months'] = df['#Passengers'].rolling(window=3).mean()
    display(df.head(10))
    # On supprime les lignes avec des valeurs manquantes
    df = df.dropna()

    # On sépare les features et la target
    X = df.drop('#Passengers', axis=1)
    #X = df[['Passengers_lag1']]
    y = df['#Passengers']

    # On sépare les données en train et test (les 24 derniers mois pour le test)
    X_train = X.iloc[:-24]
    X_test = X.iloc[-24:]

    y_train = y.iloc[:-24]
    y_test = y.iloc[-24:]

    # On crée et entraîne le modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train, y_train)

    # On fait des prédictions
    y_pred = model.predict(X_test)
    # On évalue le modèle
    rmse = np.sqrt(root_mean_squared_error(y_test, y_pred))
    print(f"RMSE sur le jeu de test: {rmse:.2f}")
    print()

    # On affiche les résultats
    plt.figure(figsize=(10, 6))
    plt.plot(y.index, y, label='Vrai', marker='o')
    plt.plot(y_test.index, y_pred, label='Prédit', marker='x')
    plt.title('Prédictions vs Réel')
    plt.xlabel('Date')
    plt.ylabel('Nombre de passagers')
    plt.legend()
    plt.show()

    input("Appuyez sur Entrée pour continuer...")

except FileNotFoundError:
    print("❌ Erreur: Le fichier 'data/AirPassengers.csv' n'a pas été trouvé.")
    print("Vérifiez que le fichier existe dans le dossier 'data'.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    sys.exit(1)
