from sklearn.preprocessing import StandardScaler 
from numpy.random import rand

X = rand(3, 4)                 # création d'une matrice aléatoire 3 x 4.
scaler = StandardScaler()      # instanciation du transformeur StandardScaler.

scaler.fit(X)                  # Calcul des espérances et écarts-types des colonnes de X et stockage dans l'objet scaler.
dir(scaler)                    # Affichage des attributs de scaler. Les attributs en question sont mean_ et scale_
                               # Vous pouvez afficher scaler.mean_ et scaler.scale_ pour voir le résultat si vous le souhaitez
Y = scaler.transform(X)        # Réduction et centrage de X.

print(scaler.mean_, scaler.scale_)

scaler2 = StandardScaler()

scaler2.fit(Y)

print(scaler2.mean_)
print(scaler2.var_)

import numpy as np
from sklearn.base import TransformerMixin  # Importation du mixin TransformerMixin
from numpy.random import rand           

class Inverseur2x2(TransformerMixin):
    def __init__(self):         
        self.det = 0                    # L'attribut det est initialisé à 0 lors de l'instanciation
    
    def fit(self, X):
        if X.shape != (2, 2):           # L'attribut shape d'un array donne les dimensions de la matrice
            print("La matrice n'est pas de dimension 2x2")
            return
        self.det = X[0,0] * X[1,1] - X[0,1] * X[1,0]    # det = ad -bc
        
        return self
        
    
    def transform(self,X):
        if self.det == 0:               # Il faut que le déterminant soit non-nul pour que la matrice soit inversible.
            print("La matrice n'est pas inversible")
            return
        Y = np.zeros([2, 2])   # Création d'une nouvelle matrice 2x2
        Y[0, 0] = X[1, 1]                              
        Y[1, 1] = X[0, 0]
        Y[0, 1] = - X[0, 1]
        Y[1, 0] = - X[1, 0]
        return (1/self.det) * Y
        
inverseur = Inverseur2x2()

X = rand(2,2)
Y = inverseur.fit_transform(X)

print(X)
print(Y)

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
poly = PolynomialFeatures(2)   # Calcule les features polynomiaux d'ordre 2 
std_scaler = StandardScaler() # Centrage et réduction
mm_scaler = MinMaxScaler()    # Normalisation min-max
union = FeatureUnion([('poly_features', poly),
                    ('standardization', std_scaler),
                    ('minmax_scaling', mm_scaler)])

print(union)