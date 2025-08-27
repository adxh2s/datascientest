from sklearn.ensemble import VotingClassifier , StackingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_validate, KFold, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# Pandas et Numpy pour dataframe et calcul
import pandas as pd
import numpy as np
# Matplotlib pour la visualisation
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch

# kds pour les métriques et visualisations
import kds

# Seaborn pour des visualisations plus esthétiques
import seaborn as sns
# Affichage dans Jupyter Notebook
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
        
        
def plot_confusion_matrix_seaborn(cm, figsize=(6,5), variable_name=None):
    """
    Affiche une matrice de confusion 2x2 avec fond coloré personnalisé et légende,
    uniquement avec Seaborn et Matplotlib.

    Paramètres :
    - cm : array-like 2x2 (ex: numpy array ou liste de listes)
        Matrice de confusion [[VN, FP], [FN, VP]]
    - figsize : tuple (largeur, hauteur)
        Taille de la figure matplotlib
    - variable_name : str ou None (optionnel)
        Nom de la variable concernée, affiché dans le titre

    Exemple d'utilisation :
    cm = np.array([[50, 5],
                   [3, 42]])
    plot_confusion_matrix_seaborn(cm, figsize=(7,6), variable_name="Classe cible")
    """
    # Labels classiques confusion matrix
    row_labels = ['Vrai Négatif', 'Faux Négatif']
    col_labels = ['Faux Positif', 'Vrai Positif']

    # Convertir en DataFrame pour seaborn (avec labels)
    df_cm = pd.DataFrame(cm, index=row_labels, columns=col_labels)
    
    # Couleurs personnalisées pour chaque catégorie
    # On crée une palette map par catégories pour colorer le fond via un DataFrame semblable
    # Vert clair pour VP/VN, rouge clair pour FP/FN
    colors_array = np.array([['#97DBAE', '#FF6978'],
                             ['#FF6978', '#97DBAE']])
    df_colors = pd.DataFrame(colors_array, index=row_labels, columns=col_labels)

    # Création de la figure
    plt.figure(figsize=figsize)

    # Heatmap avec cmap neutre et annot
    ax = sns.heatmap(df_cm, annot=True, fmt='d', cbar=False,
                     linewidths=0.8, linecolor='black',
                     square=True, cmap='Greys',
                     mask=None,
                     annot_kws={"weight": "bold", "size": 14},
                     )

    # Colore le fond des cases individuellement avec les couleurs définies (semi-transparent)
    for y in range(df_cm.shape[0]):
        for x in range(df_cm.shape[1]):
            ax.add_patch(
                plt.Rectangle((x, y), 1, 1, fill=True, facecolor=df_colors.iat[y, x], alpha=0.3, edgecolor='none', lw=0)
            )

    # Construction du titre avec variable_name
    if variable_name:
        plt.title(f"Matrice de confusion pour la variable : {variable_name}", fontsize=16)
    else:
        plt.title("Matrice de confusion", fontsize=16)

    # Légende
    legend_elements = [
        Patch(facecolor='#97DBAE', edgecolor='black', label='Vrais Positifs / Vrais Négatifs'),
        Patch(facecolor='#FF6978', edgecolor='black', label='Faux Positifs / Faux Négatifs')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1), title='Légende')

    # Labels axes
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Vérité terrain')

    plt.tight_layout()
    plt.show()

    
# Chargement des données
df = pd.read_csv('../data/diabetes.csv')

display_dataframe_info(df, "Diabetes DataFrame")

explore_column(df, 'Outcome')

# Séparation des données
# On suppose que 'Outcome' est la variable cible
data = df.drop('Outcome', axis=1)
target = df['Outcome']

# Décomposition des données en deux ensembles d'entraînement et de test
# par défaut l'échantillon est aléatoirement réparti
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=4)

# Instanciation des classifiers pour le Voting Classifier
print("🔍 Création des classificateurs pour le Voting Classifier...")
clf1 = KNeighborsClassifier(n_neighbors=5)
clf2 = RandomForestClassifier(n_estimators=200, random_state= 1)
clf3 = LogisticRegression(max_iter=1000)
clf4 = SVC(C=0.1)

# Instanciation du Voting Classifier
print("🔍 Création du Voting Classifier...")
estimators = [('knn', clf1), ('rf', clf2), ('lr', clf3)]
vclf = VotingClassifier(estimators=estimators, voting='hard')

# Cross-validation pour évaluer les classificateurs
print("🔍 Évaluation des classificateurs avec cross-validation...")
cv3 = KFold(n_splits=3, random_state=111, shuffle=True)
for clf, label in zip([clf1, clf2, clf3, vclf], ['KNN', 'Random Forest', 'Logistic Regression', 'Voting Classifier']):
    scores = cross_validate(clf, X_train, y_train, cv=cv3, scoring=['accuracy','f1'])
    print("[%s]: \n Accuracy: %0.2f (+/- %0.2f)" % (label, scores['test_accuracy'].mean(), scores['test_accuracy'].std()),
          "F1 score: %0.2f (+/- %0.2f)" % (scores['test_f1'].mean(), scores['test_f1'].std()))
    
# Grille de recherche pour optimiser les hyperparamètres
# print("🔍 Recherche des meilleurs paramètres avec GridSearchCV...")
# params = {
#             'knn__n_neighbors': [5, 9],
#             'rf__n_estimators': [20, 100, 200],
#             'svm__C': [0.01, 0.1, 1],
#             'estimators': [[('knn', clf1), ('rf', clf2), ('lr', clf3), ('svm', clf4)]] 
#           }

# # GridSearchCV pour trouver les meilleurs paramètres
# grid = GridSearchCV(estimator=vclf, param_grid=params, cv=5)
# grid = grid.fit(X_train, y_train)
# print("Meilleurs estimateur trouvés:")
# print(grid.best_estimator_)
# print("Meilleurs paramètres:")
# print(grid.best_params_)
# print("Meilleur score de validation croisée: ", grid.best_score_)

# Instanciation du Stacking Classifier
print("🔍 Création du Stacking Classifier...")
sclf = StackingClassifier(estimators=estimators, final_estimator=clf3)
# Cross-validation pour évaluer le Stacking Classifier
print("🔍 Évaluation du Stacking Classifier avec cross-validation...")
scores = cross_validate(sclf, X_train, y_train, cv=cv3, scoring=['accuracy', 'f1'])
# Affichage des scores du Stacking Classifier
print("🔍 Résultats du Stacking Classifier:")    
print("[StackingClassifier]: \n Accuracy: %0.2f (+/- %0.2f)\n" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std()),
      "F1 score: %0.2f (+/- %0.2f)" % (scores['test_f1'].mean(), scores['test_f1'].std()))

# Entraînement des classificateurs
print("📈 Entraînement des classificateurs...")
vclf.fit(X_train, y_train)
sclf.fit(X_train, y_train)

# Scores des classificateurs
print("🔍 Scores des classificateurs sur l'ensemble de test:")
print("Acc Voting   :", vclf.score(X_test, y_test))
print("Acc Stacking :", sclf.score(X_test, y_test))

# Prédictions sur l'ensemble de test
print("🔮 Prédictions sur l'ensemble de test...")
y_pred_vclf = vclf.predict(X_test)
y_pred_sclf = sclf.predict(X_test)

# Matrice de confusion Voting Classifier
print("🔍 Matrice de confusion pour le Voting Classifier:")
display(pd.crosstab(y_test, y_pred_vclf, rownames=['Classe réelle'], colnames=['Classe prédite']))
# Matrice de confusion Stacking Classifier
print("🔍 Matrice de confusion pour le Stacking Classifier:")
display(pd.crosstab(y_test, y_pred_sclf, rownames=['Classe réelle'], colnames=['Classe prédite']))
