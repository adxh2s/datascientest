import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

# Charger Iris et ne garder que 2 caractéristiques pour l'affichage
iris = datasets.load_iris()
X = iris.data[:, :2]      # longueur et largeur des sépales
y = iris.target

# Pour simplifier, ne garder que 2 classes (0 et 1)
idx = y != 2
X = X[idx]
y = y[idx]

# Séparer jeu d'entraînement et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Créer et entraîner un modèle SVM linéaire
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# Visualiser la frontière de décision
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, s=30)

# Création d’un maillage pour tracer l’hyperplan
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Tracer l'hyperplan (droit où y = 0)
plt.contour(xx, yy, Z, levels=[0], linestyles=['-'], colors='k')

# Marges (y = ±1)
plt.contour(xx, yy, Z, levels=[-1, 1], linestyles=['--'], colors='k')

plt.xlabel('Longueur sépale (x1)')
plt.ylabel('Largeur sépale (x2)')
plt.title('Frontière de décision SVM linéaire')
plt.show()
