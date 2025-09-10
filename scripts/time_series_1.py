import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
    # Insérez votre code ici
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
    data = pd.read_csv('data/AirPassengers.csv', header=0, index_col=0, parse_dates=True)
    display(data.head())
    print("✅ Données chargées avec succès!")
    print()
    

    print('Data Types:')
    print(data.dtypes)
    # Affichage riche du DataFrame
    # display_dataframe_info(df, "AirPassengers Data")
    # Exemple d'exploration d'une colonne (à décommenter si nécessaire)
    plt.plot(data)
    plt.title('Air Passengers Over Time')
    plt.show()

    # Décomposition saisonnière simple (additive)
    res = seasonal_decompose(data)
    res.plot()
    plt.title('Seasonal Decompose Additive')
    plt.show()

    # Décomposition saisonnière multiplicative
    res2 = seasonal_decompose(data, model='multiplicative')
    res2.plot()
    plt.title('Seasonal Decompose Multiplicative')
    plt.show()

    # Transformation logarithmique pour stabiliser la variance
    dflog = np.log(data)
    plt.plot(dflog)
    plt.title('Log Transformed Data')
    plt.show()

    # vignettes 2 * 2
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    # Exemple de tracés dans les sous-graphiques
    # Air Passengers Over Time
    axes[0, 0].plot(data)
    axes[0, 0].set_title('Air Passengers Over Time')
    # Log Transformed Data
    axes[0, 1].plot(np.log(data))
    axes[0, 1].set_title('Log Transformed Data')
    # Seasonal Decompose Additive
    ax = axes[1, 0]
    decomp_add = seasonal_decompose(data)
    ax.plot(decomp_add.observed, label='Observed', alpha=0.6)
    ax.plot(decomp_add.trend,    label='Trend',    alpha=0.8)
    ax.plot(decomp_add.seasonal, label='Seasonal', alpha=0.8)
    ax.plot(decomp_add.resid,    label='Residual', alpha=0.6)
    ax.set_title('Seasonal Decompose (Additive)')
    ax.legend(loc='upper right', fontsize=8)
    # Seasonal Decompose Multiplicative
    ax = axes[1, 1]
    decomp_mul = seasonal_decompose(data, model='multiplicative')
    ax.plot(decomp_mul.observed, label='Observed', alpha=0.6)
    ax.plot(decomp_mul.trend,    label='Trend',    alpha=0.8)
    ax.plot(decomp_mul.seasonal, label='Seasonal', alpha=0.8)
    ax.plot(decomp_mul.resid,    label='Residual', alpha=0.6)
    ax.set_title('Seasonal Decompose (Multiplicative)')
    ax.legend(loc='upper right', fontsize=8)
    # Ajustement de l'espacement entre les sous-graphiques
    plt.tight_layout()
    # Affichage du graphique
    plt.show()

    # On applique la fonction seasonal_decompose à airpasslog 
    # Ici on utilise la transformée en log pour avoir un modèle additif
    airpass = data
    display(airpass.index)
    # On passe une fonction log pour stabiliser la variance
    airpasslog = np.log(airpass)
    display(airpasslog)
    # On applique la décomposition additive
    mult = seasonal_decompose(airpasslog)
    # On soustrait les coefficients saisonniers à la série airpasslog
    cvs = airpasslog['#Passengers'] - mult.seasonal
    # On applique la fonction exponentielle pour retrouver la série originale
    x_cvs = np.exp(cvs)
    # On affiche la série
    plt.plot(airpass, label='Série originale')
    plt.plot(x_cvs, label='Série corrigée') 

    plt.title('Graphique 1 de la série originale et la série corrigée')
    plt.xlabel('Date')
    plt.ylabel('Nb passagers')
    plt.legend()
    plt.show()


    mult2 = seasonal_decompose(airpass, model='multiplicative')
    # On soustrait les coefficients saisonniers à la série airpasslog
    cvs2 = airpass['#Passengers'] / mult2.seasonal
    # On affiche la série
    plt.plot(airpass, label='Série originale')
    plt.plot(cvs2, label='Série corrigée') 

    plt.title('Graphique 2 de la série originale et la série corrigée')
    plt.xlabel('Date')
    plt.ylabel('Nb passagers')
    plt.legend()
    plt.show()

    # Moyenne mobile centrée
    # Exercice :
    # • (a) A l'aide de la méthode rolling combinée avec mean, calculer la moyenne mobile centrée sur 12 mois et la stocker dans une variable appelée airpass_ma.
    # • (b) Représenter dans un même graphique la moyenne mobile airpass_ma et la série airpasslog.
    # • (c) Donner un titre à votre graphique, une légende à vos courbes et des labels à vos axes.
    
    # (a) Calcul de la moyenne mobile centrée sur 12 mois
    airpass_ma = airpasslog.rolling(window=12, center=True).mean()
    display(airpass_ma.head(15))
    # (b) Représentation graphique
    plt.plot(airpass_ma, color='red', label='Série moyenne mobile')
    plt.plot(airpasslog, color='blue', label='Série log corrigée') 
    # (c) Ajout des titres et légendes
    plt.title('Graphique de la série moyenne mobile et la série corrigée')
    plt.xlabel('Date')
    plt.ylabel('Nb passagers')
    plt.legend()
    plt.show()

    # On repasse en échelle normale   
    plt.plot(np.exp(airpasslog), color = 'blue', label = 'Origine')
    plt.plot(np.exp(airpass_ma), color = 'red', label = 'Moyenne mobile')
    plt.legend()
    plt.title('Méthode des moyennes mobiles')
    plt.show()

    # Recherche de la saisonnalité
    #  Calcul de la différence entre la série 𝑋𝑡 et sa moyenne mobile M12𝑋𝑡.
    airpasslog_without_ma = airpasslog - airpass_ma
    display(airpasslog_without_ma.head(10))
    # On enlève les valeurs NaN
    airpasslog_without_ma = airpasslog_without_ma.dropna()
    airpasslog_without_ma.head(10)
    # On ajoute une colonne month
    df=airpasslog_without_ma
    df['month'] = (df.index).month
    df.head(12)
    # On calcule la saisonnalité
    # Groupby month et on fait la moyenne
    seasonality = df.groupby('month').mean()
    display(seasonality.head(12))
    # On centre la saisonnalité
    seasonality = seasonality - seasonality.mean()
    display(seasonality.head(12))
    # Création du vecteur 
    seasonal_vector = np.zeros(144)

    j=0
    for i in range(12):
        seasonal_vector[j:j+12] = seasonality.iloc[:,0]
        j=j+12
        
    # Soustraire les coefficients saisonniers
    airpasslog_cvs = airpasslog["#Passengers"] - seasonal_vector
    # On retrouve la série originale par passage à l'exponentielle
    airpass_cvs = np.exp(airpasslog_cvs)
    
    # On affiche la série originale et la série corrigée
    plt.plot(airpass, ':', label = '$X_t$')
    plt.plot(airpass_cvs, label = '$X_t^{CVS}$')
    plt.title('Série originale et corrigée de ses variations saisonnières')
    plt.xlabel('t')
    plt.ylabel('# Passengers')
    plt.legend()
    plt.show()


    input("Press Enter to continue...")
except FileNotFoundError:
    print("❌ Erreur: Le fichier 'data/AirPassengers.csv' n'a pas été trouvé.")
    print("Vérifiez que le fichier existe dans le dossier 'data'.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    sys.exit(1)
