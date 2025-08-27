import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    # df = pd.read_csv('../data/marvel-wikia-data.csv')
    df = pd.read_csv('../data/CarPrice_Assignment.csv')
    print("✅ Données chargées avec succès!")
    print()
    
    # Affichage riche du DataFrame
    display_dataframe_info(df, "CarPrice Data")
    # Exemple d'exploration d'une colonne (à décommenter si nécessaire)
    explore_column(df, 'symboling')  # Remplacez 'name' par le nom d'une colonne de votre DataFrame
    
    # Variables explicatives
    X = df.drop(['car_ID', 'CarName', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'price'], axis = 1)
    # Variable cible
    y = df['price']
    
    # Affichage des variables explicatives
    print("Affichage des variables explicatives (X) \n",X.head())

    # Affichage de la variable cible
    print("Affichage de la variable cible (y) \n",y.head())
    
    # entrainement du model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 42)
    
    # Instanciation du modèle
    lr = LinearRegression()      

    # Entraînement du modèle sur le jeu d'entraînement
    lr.fit(X_train, y_train)        

    # Prédiction de la variable cible pour le jeu de données d'entrainement. Ces prédictions sont stockées dans y_pred_train
    y_pred_train = lr.predict(X_train)         

    # Prédiction de la variable cible pour le jeu de données test. Ces prédictions sont stockées dans y_pred_test
    y_pred_test = lr.predict(X_test)    
    
    # Affichage des prédictions
    print("Affichage des prédictions Train \n",y_pred_train)
    print("Affichage des prédictions Test \n",y_pred_test)
    
    #Affichage des erreurs
    print("Affichage des erreurs Train \n",y_pred_train - y_train)
    print("Affichage des erreurs Test \n",y_pred_test - y_test)
    
    # Affichage des coefficients
    print("Affichage des coefficients \n",lr.coef_)
    print("Affichage de l'intercept \n",lr.intercept_)
    
    #Evaluation du modele
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    print("Affichage de l'erreur quadratique moyenne (train) \n",mse_train)
    print("Affichage de l'erreur quadratique moyenne (test) \n",mse_test)

    #Affichage de l'erreur absolue moyenne
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    print("Affichage de l'erreur absolue moyenne (train) \n",mae_train)
    print("Affichage de l'erreur absolue moyenne (test) \n",mae_test)

    #Affichage de l'erreur relative
    mean_price = df['price'].mean()
    print("Affichage de la moyenne des prix \n",mean_price)
    print("Affichage de l'erreur relative", mae_test / mean_price)

except FileNotFoundError:
    print("❌ Erreur: Le fichier '../data/marvel-wikia-data.csv' n'a pas été trouvé.")
    print("Vérifiez que le fichier existe dans le dossier 'data'.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    sys.exit(1)
