import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_digits
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

X = pd.DataFrame(load_diabetes().data)
y = pd.DataFrame(load_diabetes().target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 1)

scaler = StandardScaler()          # Transformeur : Normalisation
poly = PolynomialFeatures(2)       # Transformeur : Features polynomiaux
linreg = LinearRegression()        # Modèle : Régression linéaire


linreg_pipe = Pipeline(steps = [('normalization', scaler),            # Etape 1 : Normalisation
                                 ('poly_features', poly),             # Etape 2 : Calcul features polynomiaux
                                 ('linear_regression', linreg)])      # Etape 3 : Régression Linéaire

linreg_pipe.fit(X_train, y_train)                # Entrainement de la Pipeline

score_pipe = linreg_pipe.score(X_test, y_test)

linreg.fit(X_train, y_train)                     # Entrainement d'une régression linéaire classique avec paramètres par défaut
score_classique = linreg.score(X_test, y_test)

print("le score de la pipeline est: ", score_pipe)
print("le score de la régression linéaire classique est:", score_classique)



X = pd.DataFrame(load_digits().data)
y = load_digits().target

#Nous retirons les variables constantes dans la ligne suivante :
X = X.drop([0, 24, 31, 32, 39, 56], axis=1)


scaler = StandardScaler()  # Transformeur : Normalisation
selector = SelectKBest()   # Transformeur : Sélection de variables
svc = SVC()                # Modèle : Support Vector Classifier

svc_pipe = Pipeline([ ('scaling', scaler),       # Etape 1 : Normalisation des données
                      ('selection', selector),   # Etape 2 : Sélection des k meilleures variables
                      ('model', svc)])           # Etape 3 : Entrainement d'un modèle SVC

param_grid = {
    'selection__k' : [10, 20, 30, 40, 50, 'all'],  # On teste avec 10, 20, ... , 50 et toutes les variables de X
    'model__kernel' : ['poly', 'linear', 'rbf']    # On teste SVC avec noyau polynomial, linéaire et gaussien
}

grid = GridSearchCV(estimator = svc_pipe, param_grid = param_grid, cv = 5) # Instanciation d'une GridSearchCV

grid.fit(X,y)  # Entraînement de la GridSearchCV pour trouver les meilleurs paramètres de la pipeline.

print(grid.best_params_) # Affichage des meilleurs paramètres pour la pipeline svc_pipe.
print(grid.best_score_) # Affichage du score obtenu par validation croisée avec les meilleurs paramètres.

from sklearn.base import BaseEstimator, TransformerMixin

class ColumnBucketer(BaseEstimator, TransformerMixin):
    # BaseEstimator contient les méthodes get_params et set_params.
    # TransformerMixin contient la méthode fit_transform.

    def __init__(self, column_name, bucket_size):
        self.column_name = column_name   # nom de la colonne à segmenter
        self.bucket_size = bucket_size   # longueur de chaque segment

    def fit(self, X, y):  # Ne fait rien
        return self

    def transform(self, X):  # Création de la nouvelle colonne
        X[self.column_name + "_Bucket"] = X[self.column_name] // self.bucket_size * self.bucket_size
        return X


import warnings
warnings.filterwarnings('ignore')


from sklearn.datasets import fetch_california_housing

import pandas as pd
import numpy as np

housing = fetch_california_housing()
print(housing, type(housing))
# X = pd.DataFrame(load_boston().data, columns= load_boston().feature_names)
# y = load_boston().target

# X = pd.DataFrame(load_boston().data, columns= load_boston().feature_names)
X = pd.DataFrame(housing['data'], columns=housing['feature_names'])
y = housing['target']

print(X.shape)
print(y.shape)

class ColumnDrop(BaseEstimator, TransformerMixin):

    def __init__(self, column_to_drop):
        self.column_to_drop = column_to_drop   # nom de la colonne à supprimer

    def fit(self, X, y):  # Ne fait rien
        return self

    def transform(self, X): # Suppression de la colonne
        return X.drop(self.column_to_drop, axis = 1)

from sklearn.linear_model import BayesianRidge

## Instanciation des transformeurs de la pipeline
segmentation_age = ColumnBucketer(column_name = 'HouseAge', bucket_size = 5)
column_drop_age =  ColumnDrop(column_to_drop = 'HouseAge')

## Instanciation du modèle
byrdge = BayesianRidge()

## Définition de la pipeline
byrdge_pipe = Pipeline(steps = [('bucket_age', segmentation_age),  # Etape 1 : Segmentation de la colonne 'AGE'
                                # ('bucket_crim', segmentation_crim),# Etape 2 : Segmentation de la colonne 'CRIM'
                                ('drop_age', column_drop_age),     # Etape 3 : Suppression de la colonne 'AGE'
                                # ('drop_crim', column_drop_crim),   # Etape 4 : Suppression de la colonne 'CRIM'
                                ('model', byrdge)                  # Etape 5 : Entrainement d'un modèle BayesianRidge
                               ])

params = {
    'bucket_age__bucket_size' : [3, 5, 10],      # variation du paramètre bucket_size de l'étape bucket_age
    # 'bucket_crim__bucket_size' : [3, 5, 10],     # variation du paramètre bucket_size de l'étape bucket_crim
    'model__alpha_1' : np.logspace(-3, 2, 5),   # variation du paramètre alpha_1 de l'étape model
    'model__alpha_2' : np.logspace(-3, 2, 5)    # variation du paramètre alpha_2 de l'étape model
}

grid = GridSearchCV(byrdge_pipe, param_grid = params, cv = 5)

grid.fit(X,y)

print('Le meilleur score obtenu est: ',grid.best_score_)
print('Les meilleurs paramètres trouvés sont :',grid.best_params_)

# # exemple de columnTransformer et de la fonction eponyme
# from sklearn.compose import ColumnTransformer
# from sklearn.compose import make_column_transformer
# # Standardisation des valeurs
# numerical_st = StandardScaler()
# # Encodage des variables catégorielles
# categorical_ohe = OneHotEncoder()
# #num_vars : liste des variables numériques
# #cat_vars : liste des variables catégorielles
# prepro = ColumnTransformer(

#     transformers = [('numerical', numerical_st, num_vars),
#                     ('categorical', categorical_ohe, cat_vars)])

# # OU --> mais pas de nom de transformer pour les passer à un grid search !!!
# prepro = make_column_transformer((numerical_st, num_vars), #num_var : liste des variables numériques
#                             (categorical_ohe, cat_var)) #cat_var : liste des variables catégorielles