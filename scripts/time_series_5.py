import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import sys
import os
from IPython.display import display
    # Insérez votre code ici



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
    df = pd.read_csv('data/AirPassengers.csv', parse_dates=True)
    print("✅ df head")
    display(df.head())
    print("✅ Données chargées avec succès!")
    print()
    
    # Renommer les colonnes pour Prophet
    df = df.rename(columns={"Month": "ds", "#Passengers": "y"})
     print("✅ df rename ")
    display(df.head())

    # Séparation train et test selon la chronologie
    df_train = df.iloc[:-24]
    df_test = df.iloc[-24:]

    # instanciation modele Prophet et ajustement aux données train
    model = Prophet()
    model.fit(df_train)

    # données futures sur 24 mois
    future = model.make_future_dataframe(periods=24, freq='ME')
     print("✅ future")
    display(future.head())
    display(future.tail())
    forecast = model.predict(future)

    print("✅ Forecast")
    model.plot(forecast)
    display(forecast)
    model.plot_components(fcst=forecast)

    from sklearn.metrics import root_mean_squared_error

    rmse = root_mean_squared_error(df_test['y'], forecast['yhat'].tail(24))
    print(f'Erreur quadratique moyenne (RMSE) : {rmse}')

    plt.figure(figsize=(10, 6))
    plt.plot(df_test['ds'], df['y'].tail(24), label='Valeurs réelles')
    plt.plot(df_test['ds'], forecast['yhat'].tail(24), label='Prédictions', linestyle='--')
    plt.fill_between(df_test['ds'], forecast['yhat_lower'].tail(24), forecast['yhat_upper'].tail(24), 
                    color='orange', alpha=0.15, label='Intervalle de confiance (80%)')
    plt.title('Prédictions de Prophet')
    plt.xlabel('Date')
    plt.xticks(df_test['ds'], rotation=45)
    plt.ylabel('Nombre de passagers')
    plt.legend()
    plt.show()

    param_fixed = {  
        'seasonality_mode': 'multiplicative',
        'daily_seasonality': False,
        'weekly_seasonality': False,
        'yearly_seasonality': True
    }

    param_grid = {  
        'changepoint_prior_scale': [0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.1, 1, 10, 15],
        'changepoint_range': [0.5, 0.75, 0.95]
    }

    import itertools
    import numpy as np
    from prophet.diagnostics import cross_validation
    from prophet.diagnostics import performance_metrics


    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store the RMSEs for each params here

    # Use cross validation to evaluate all parameters
    for params in all_params:
        model_cv = Prophet(**param_fixed, **params).fit(df_train)  # Fit model with given params
        df_cv = cross_validation(model_cv, initial='2920 days', period='365 days', horizon = '365 days', parallel="threads")
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])

    # Find the best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    display(tuning_results)

    best_params = all_params[np.argmin(rmses)]
    best_score = tuning_results['rmse'].min()
    print(best_params)
    print(best_score)

    model_tuned = Prophet(**best_params, **param_fixed)
    model_tuned.fit(df_train)
    predictions_tuned = model_tuned.predict(future)

    model_tuned.plot(predictions_tuned)
    model_tuned.plot_components(predictions_tuned)

    rmse = root_mean_squared_error(df_test['y'], predictions_tuned['yhat'].tail(24))
    print(f'Erreur quadratique moyenne (RMSE) : {rmse}')

    plt.figure(figsize=(10, 6))
    plt.plot(df_test['ds'], df['y'].tail(24), label='Valeurs réelles')
    plt.plot(df_test['ds'], predictions_tuned['yhat'].tail(24), label='Prédictions', linestyle='--')
    plt.fill_between(df_test['ds'], predictions_tuned['yhat_lower'].tail(24), predictions_tuned['yhat_upper'].tail(24), 
                    color='orange', alpha=0.15, label='Intervalle de confiance (80%)')
    plt.title('Prédictions du modèle Prophet optimisé')
    plt.xlabel('Date')
    plt.xticks(df_test['ds'], rotation=45)
    plt.ylabel('Nombre de passagers')
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
