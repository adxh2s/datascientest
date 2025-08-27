import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Génération d'un jeu de données complexe (4 groupes, inégalement répartis)
np.random.seed(42)
X = np.vstack([
    np.random.randn(60, 2) + [0, 0],
    np.random.randn(15, 2) + [5, 5],
    np.random.randn(30, 2) + [10, 0],
    np.random.randn(10, 2) + [12, 8]
])

# Calcul de la matrice de liaison avec méthode 'ward'
Z = linkage(X, method='ward')

# Affichage du dendrogramme avec visualisation des hauteurs potentielles
plt.figure(figsize=(12, 6))
dn = dendrogram(Z)

# Tracer plusieurs lignes horizontales pour indiquer différentes hauteurs à tester
hauteurs_tests = [5, 7, 10]

for h in hauteurs_tests:
    plt.axhline(y=h, color='red', linestyle='--', label=f'Hauteur testée = {h}')

plt.title("Dendrogramme complexe avec hauteurs testées")
plt.xlabel("Index des observations")
plt.ylabel("Hauteur / distance de fusion")
plt.legend()
plt.show()

# Fonction pour découper le dendrogramme à une hauteur et afficher la répartition
def plot_clusters_from_height(Z, X, hauteur):
    labels = fcluster(Z, t=hauteur, criterion='distance')
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10')
    plt.title(f'Clustering pour coupure dendrogramme à hauteur {hauteur}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    print(f"Nombre de clusters à hauteur {hauteur} : {len(np.unique(labels))}")
    print(f"Répartition des clusters : {np.bincount(labels)[1:]}")

# Tester les coupes à différentes hauteurs
for h in hauteurs_tests:
    plot_clusters_from_height(Z, X, h)
