# model learning regression lineaire
from sklearn import linear_model, preprocessing 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Pour l'encodage des variables catégorielles
from sklearn.preprocessing import OneHotEncoder

# Pandas et Numpy pour dataframe et calcul
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

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
        
        
# chargement donnnées
df = pd.read_csv('../data/Admission_Predict.csv', index_col=0)
display(df.head())

# Informations du dataframe
display_dataframe_info(df)

# On renomme les colonnes pour éviter les espaces
df = df.rename(columns={
                        "GRE Score": "GRE",
                        "TOEFL Score": "TOEFL",
                        "University Rating": "URATE",
                        "Research": "RESEARCH",
                        "Chance of Admit ": "ADMIT"
                        }
)

# Affichage des informations du DataFrame après renommage
display_dataframe_info(df, title="DataFrame après renommage des colonnes")

# 2 - Exploration de colonnes spécifiques
# display(df["Serial No."].value_counts())
# display(df["Serial No."].nunique())
# --> On met cette colonne en index pour qu'elle ne perturbe pas les analyses
# df.set_index("Serial No.", inplace=True) ou read_csv avec index_col=0
explore_column(df, "ADMIT")
# groupe d'élèves par admission / moyenne gre
seuil_admit = 0.73
# Création d’une colonne de regroupement sur admission, basé sur le seuil = mediane
df['GRPADMIT'] = df['ADMIT'].apply(lambda x: '< seuil' if x < seuil_admit else '>= seuil')
# Groupby puis moyenne sur 'scores'
gp_gre = df.groupby('GRPADMIT')['GRE'].mean()
print(gp_gre)

# fonction cut pour discretionner gre
gre_discret = pd.cut(df['GRE'], bins=[200, 300, 320, 330, 400],
                     labels=['mauvais', 'moyen', 'bon','excellent']
)
crosstab = pd.crosstab(df['GRPADMIT'], gre_discret)
display(crosstab)
# On ajoute la colonne de discrétisation à notre DataFrame
df = df.merge(gre_discret, left_index=True, right_index=True)
display(df)

df = df.drop(columns='GRE_x')
df = df.rename(columns={'GRE_y': 'GRE'})

# Exploration de la colonne TOEFL
explore_column(df, "TOEFL")

# groupe d'élèves par admission / moyenne toefl
gp_toefl = df.groupby('GRPADMIT')['TOEFL'].mean()
print(gp_toefl)

# On crée une colonne de discrétisation pour TOEFL
toefl_discret = pd.cut(x=df['TOEFL'],
                       bins=[90, 105, 110, 115, 130],
                       labels=['mauvais', 'moyen', 'bon', 'excellent']
)

crosstab = pd.crosstab(df['GRPADMIT'], toefl_discret, normalize='columns') # ou normalize = 1
display(crosstab)
# On ajoute la colonne de discrétisation à notre DataFrame
df = df.merge(toefl_discret, left_index=True, right_index=True)
display(df)

df = df.drop(columns='TOEFL_x')
df = df.rename(columns={'TOEFL_y': 'TOEFL'})

# fonction cut pour discretionner admit
admit_discret = pd.cut(df['ADMIT'], bins=[0, 0.73, 1],
                     labels=[0, 1]
)
# On ajoute la colonne de discrétisation à notre DataFrame
df = df.merge(admit_discret, left_index=True, right_index=True)
display(df)

df = df.drop(columns='ADMIT_x')
df = df.rename(columns={'ADMIT_y': 'ADMIT'})
display(df)

# Séparation des données en features et target
data = df.drop(columns=['ADMIT', 'GRPADMIT'], axis='1')
target = df['ADMIT']

# Affichage des données et de la cible
display(data)
display(target)

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=66)

# Encodage des variables catégorielles

enc = OneHotEncoder(handle_unknown='ignore')

#Adaptez l'encodeur aux données d'entraînement (X_train) et transformez simultanément X_train 
X_train_enc = enc.fit_transform(X_train)

#Transformez également les données de test (X_test). 
#Pas besoin d'adapter l'encodeur aux données de test car il a déjà été adapté aux données d'entraînement.
#Utiliser simplement la méthode transform(). 
X_test_enc = enc.transform(X_test)

# Affichage des données et de la cible
display(X_train_enc)
display(X_test_enc)

# Instanciation de la classe LogisticRegression
clf = linear_model.LogisticRegression(C=1.0)

# Entrainement du model learning
clf.fit(X_train_enc, y_train)

y_pred = clf.predict(X_test_enc)

# Calcul de la matrice de confusion 

## Méthode 1 : à l'aide de sklearn

cm = confusion_matrix(y_test, y_pred)
print(cm)

## Méthode 2 : à l'aide de pandas
cm2 = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
print(cm2)

# score du modèle
print(clf.score(X_test_enc, y_test))

from  sklearn.metrics import classification_report

print(classification_report(y_test,  y_pred))


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Calcul des probabilités de prédiction
# On utilise predict_proba pour obtenir les probabilités de chaque classe
probs = clf.predict_proba(X_test_enc)
print(probs)
# On considère que la classe 1 est positive et on applique un seuil de 0.4
# pour décider si on prédit 1 ou 0
y_preds = np.where(probs[:,1]>0.4,1,0)
print(y_preds)
# Calcul de la matrice de confusion avec les prédictions
cm = confusion_matrix(y_test,y_preds)
print(cm)

# Définition de l’ordre des classes
labels = ['Non Admis', 'Admis']

# Affichage avec les labels sur les axes
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.xlabel("Valeurs prédites")
plt.ylabel("Valeurs vraies")
plt.title("Matrice de confusion avec labels explicites")
plt.show()

from sklearn.metrics import roc_curve, auc
fpr, tpr, seuils = roc_curve(y_test, probs[:,1], pos_label=1)
roc_auc = auc(fpr, tpr)
print(fpr, "\n", tpr, "\n", seuils, "\n", roc_auc)

import matplotlib.pyplot as plt

plt.plot(fpr, tpr, color='orange', lw=2, label='Modèle clf (auc = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire (auc = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux faux positifs')
plt.ylabel('Taux vrais positifs')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.show()