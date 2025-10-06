
# (a) Charger les bibliothèques pandas, numpy, matplolib.pyplot et statsmodels sous les noms pd, np, plt et sm.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm    

# Ajout en local
from IPython.display import display

# (b) Charger les données contenues dans le fichier portland_v2.csv. Faire attention au format de l'index et au type des données avant de passer aux questions suivantes. Vous pouvez vérifier que la série est bien indexée par des données calendaires en affichant l'attribut index de la série.

# df = pd.read_csv('data/portland_timeseries.csv', index_col=0, parse_dates=True)
df = pd.read_csv('data/total_rides_2.csv') #, index_col=0, parse_dates=True)
display(df.head())
display(df.info())
display(df.index)

# Regroupement par date et somme sur les riders
df = df.groupby('date', as_index=False)['riders'].sum()
# Changement du type de l'index en DatetimeIndex
df.index = pd.to_datetime(df.index)
# Conversion de la colonne date au format datetime
df['date'] = pd.to_datetime(df['date'])
# Conversion de la colonne riders au format int64
df['riders'] = df['riders'].astype('int64')
# Mise en place de la colonne date en index
df = df.set_index('date')

display(df.head())
display(df.info())

# (c) Afficher la série entière sur un graphique.
plt.plot(df)
plt.title('Copenhagen Bike Riders Over Time')
plt.show()



# (d) Effectuer deux décompositions saisonnières à l'aide de statsmodels : 
# la première avec un modèle additif
from statsmodels.tsa.seasonal import seasonal_decompose

# Décomposition saisonnière simple (additive)
add_seasonal_dec = seasonal_decompose(df)
add_seasonal_dec.plot()
# Afficher les graphiques correspondants.
plt.title('Seasonal Decompose Additive')
plt.show()
# On voit que les résidus ne sont pas stables --> ce n'est pas un modele additif

# la seconde avec un modèle multiplicatif. 
# Décomposition saisonnière multiplicative
mul_seasonal_dec = seasonal_decompose(df, model='multiplicative')
mul_seasonal_dec.plot()
# Afficher les graphiques correspondants.
plt.title('Seasonal Decompose Multiplicative')
plt.show()

# (e) Y'a-t-il une saisonnalité ? Si oui, de quelle période ?
# On voit une saisonnalité de 12 mois (periodes sur 24 mois)
plt.plot(mul_seasonal_dec.seasonal.iloc[7:32])
plt.show()
# On voit que les résidus sont bien plus stables --> on voit également une saisonnalité

# (f) On cherche à choisir le modèle qui donne les résidus les plus stationnaires. Quel est ce modèle ?
# Modele multiplicatif

# g) A l'aide de Numpy, stocker dans la variable datalog le logarithme de la série. Afficher la nouvelle série sur un graphique. Pourquoi cette manipulation est-elle pertinente dans notre cas ?
# Transformation logarithmique
datalog = np.log(df)
plt.plot(datalog)
plt.title('Log Transformed Data')
plt.show()

# Elle permet de lisser les effets multiplicatifs et 
# on peut facilement revenir sur la serie initiale avec l'exponentielle


# (h) Afficher l'autocorrélogramme de la série datalog. Faut-il différencier la série ? Pourquoi ?
# Autocorrélation
pd.plotting.autocorrelation_plot(datalog)
plt.title('Autocorrelation Plot')
plt.show()

# Oui, il faut différencier la série, 
# car la courbe n'est pas dans la zone des bornes acceptables et ne tend pas vers 0 unifiormément

     
# (i) Créer et afficher l'autocorrélogramme de la série datalog_1, correspondant à la série datalog différienciée à l'ordre 1. 
# On pensera à supprimer les valeurs manquantes créées par la différentiation. 
# La série semble-t-elle stationnaire ?

# première différentiation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7)) 
 # Différenciation ordre 1
datalog_1 = datalog.diff().dropna()
# Série temporelle différenciée
datalog_1.plot(ax = ax1) 
ax1.set_title('Série différenciée d\'ordre 1')
# Autocorrélogramme de la série différenciée
pd.plotting.autocorrelation_plot(datalog_1, ax = ax2)
ax2.set_title('Autocorrélogramme de la série différenciée d\'ordre 1')
plt.show()

# on voit que la serie décroit vers 0, mais il reste des éléments marquants en première partie de graphique (pics saisonniers)

# j) Créer et afficher l'autocorrélogramme de la série datalog_2, correspondant à la série datalog_1 différenciée et désaisonnalisée. 
# La série semble-t-elle stationnaire ?

# seconde différentiation, avec un ordre 12 (periode saisonnière)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7)) 
 # 2eme Différenciation d'ordre 12
datalog_2 = datalog_1.diff(periods=12).dropna()
# Série temporelle doublement différenciée
datalog_2.plot(ax = ax1) 
ax1.set_title('Série doublement différenciée')
# Autocorrélogramme de la série différenciée
pd.plotting.autocorrelation_plot(datalog_2, ax = ax2)
ax2.set_title('Autocorrélogramme de la série doublement différenciée')
plt.show()

# on voit que la serie tend plus vite vers 0 et que la courbe est bien dans les bornes de l'autocorrélation, 
# sans effets saisonniers marqués, la serie semble stationnaire

# (k) Utiliser le test augmenté de Dickey-Fuller (ADF), 
# implémenté dans la libraire statsmodels à travers la fonction adfuller du sous-module tsa.stattools. 
# Conclure sur la stationnarité de datalog_2.

# Test de Dickey-Fuller augmenté pour vérifier la stationnarité
_, p_value, _, _, _, _  = sm.tsa.stattools.adfuller(datalog_2)
print(p_value)  # p-valeur bien inférieure à 0.05, on peut considérer la série comme stationnaire.

# La série est bien stationnaire

# l) Utiliser les fonctions plot_acf et plot_pacf du sous module statsmodels.graphics.tsaplots pour tracer l'autocorrélogramme simple (ACF) et le partiel (PACF) de la série datalog_2. Fixer lags à 36.

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

# Trace des fonctions d'autocorrélation et d'autocorrélation partielle
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7))
# autocorrélation
plot_acf(datalog_2, lags = 36, ax=ax1)
ax1.set_title("Fonction d'autocorrélation (ACF)")
# autocorrélation partielle
plot_pacf(datalog_2, lags = 36, ax=ax2)
ax2.set_title("Fonction d'autocorrélation partielle (PACF)")
# Affichage
plt.show()

# m) A l'aide des graphiques obtenus, déterminer visuellement les ordres p, q, P, Q.

# On rappelle ici les règles à suivre pour déterminer les ordres.

# Pour la partie non-saisonnière (p et q) :

# Ordre p - regarder PACF :

#     Compter les pics significatifs jusqu'au premier "trou" (le premier pic ne compte pas, il fait seulement office de référence)
#     Les pics doivent dépasser les lignes pointillées (intervalles de confiance)
#     p = nombre de pics significatifs consécutifs

# Ordre q - regarder ACF :

#     Même principe : compter les pics significatifs jusqu'au premier "trou"
#     q = nombre de pics significatifs consécutifs

# Pour la partie saisonnière (P et Q) :

# Ordre P - regarder PACF :

#     Observer les pics aux multiples de s=12
#     Compter les pics saisonniers significatifs (12, 24, 36...)
#     P = nombre de pics saisonniers significatifs

# Ordre Q - regarder ACF :

#     Observer les pics aux multiples de s=12
#     Compter les pics saisonniers significatifs
#     Q = nombre de pics saisonniers significatifs

# p,q 1,1 (11 mois)
# P,Q 1,1 (12 mois)

# (n) Instancier un modèle SARIMAX sur la série datalog. 
# On prendra arbitrairement les paramètres p=0, d=1, q=0, P=0, D=1, Q=1, et k=12. 
# Afficher une analyse du modèle à l'aide de la méthode summary.
model=sm.tsa.SARIMAX(datalog,order=(0,1,0),seasonal_order=(0,1,1,12))
sarima=model.fit()
print(sarima.summary())


# (o) A l'aide de la méthode get_forecast, prédire les valeurs de la série sur les 12 mois suivant la dernière valeur. 
# Afficher les prédictions et la série originale sur un même graphe. 
# Penser au log appliqué aux questions précédentes.
prediction = sarima.get_forecast(steps =12).summary_frame()  #Prédiction avec intervalle de confiance
# Préparation du graphique
fig, ax = plt.subplots(figsize = (15,5))
# affichage de la série initiale
plt.plot(df)
prediction = np.exp(prediction) # Passage par l'exponentielle pour revenir sur les valeurs initiales
# Moyenne prédite
prediction['mean'].plot(ax = ax, style = 'k--')
# Intervalle de confiance
ax.fill_between(prediction.index, prediction['mean_ci_lower'], prediction['mean_ci_upper'], color='k', alpha=0.1)
plt.title('Prévisions SARIMA pour les 12 prochains mois')
plt.show()


# (p) Afficher sur un même graphique les prédictions, les valeurs réelles, et si vous le souhaitez, l'intervalle de confiance des prédictions.
# df_12 = pd.read_csv('portland_8182.csv', index_col=0, parse_dates=True)
# display(df_12.index)
# # Préparation du graphique
# fig, ax = plt.subplots(figsize = (15,5))
# # affichage de la série initiale
# plt.plot(df_12)
# # Moyenne prédite
# prediction['mean'].plot(ax = ax, style = 'k--')
# # Intervalle de confiance
# ax.fill_between(prediction.index, prediction['mean_ci_lower'], prediction['mean_ci_upper'], color='k', alpha=0.1)
# plt.title('Prévisions SARIMA pour les 12 prochains mois')
# plt.show()




# On définit l'erreur de prédiction comme suit : $$X-\widehat{X}$$ Elle permet de savoir si la prédiction surévalue ou sous-évalue les valeurs réelles de la série.
# On définit également l'erreur moyenne relative : $$\displaystyle 100 \cdot \overline{\frac{|X - \widehat{X}|}{X}}$$ 
# Elle permet d'évaluer la qualité des prédictions : plus le pourcentage est faible, plus les précisions collent à la réalité.
# (q) Calculer l'erreur de prédiction et l'erreur moyenne relative de la prédiction.
# (r) Conclure sur la qualité du modèle. Sous-évalue-t-il ou surévalue-t-il les données ?

# Inverse log si le modèle a été entraîné sur log(y)
yhat = prediction['mean']
ci_lower = prediction['mean_ci_lower']
ci_upper = prediction['mean_ci_upper']

# Vérités terrain alignées (mêmes dates que les prévisions)
y_true = df.loc[yhat.index].squeeze()

# (p) Erreur de prédiction point à point
err = y_true - yhat  # >0 sous-évaluation, <0 surévaluation

# (q) Erreur moyenne relative (MAPE en %)
mape = np.mean(np.abs(err / y_true)) * 100
# Alternative robuste:
# from sklearn.metrics import mean_absolute_percentage_error
# mape = mean_absolute_percentage_error(y_true, yhat) * 100

print(f"Biais moyen: {err.mean():.3f}")
print(f"MAPE: {mape:.2f}%")

# (option) Visualisation
fig, ax = plt.subplots(figsize=(15,5))
df.plot(ax=ax, label='Observé')
yhat.plot(ax=ax, label='Prévu')
ax.fill_between(yhat.index, ci_lower, ci_upper, color='C1', alpha=0.2, label='IC 95%')
ax.legend()





