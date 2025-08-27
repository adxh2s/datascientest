import matplotlib.pyplot as plt
import numpy as np

## Données
# x = [1, 2, 3, 4, 5]
# y = [2, 4, 6, 8, 10]

# ## Crée un graphique
# plt.plot(x, y)

# ## Ajoute des étiquettes et un titre
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Simple Plot')

# ## Affiche le graphique
# plt.show()

# # Exemple de tracé simple
# x = [1, 2, 3, 4]
# y = [1, 4, 9, 16]
# plt.plot(x, y, 'ro-')
# plt.xlabel('Axe X')
# plt.ylabel('Axe Y')
# plt.title('Exemple de graphique')
# plt.show()

## Données
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

## Crée un graphique
plt.plot(x, y, color='red', linewidth=2, linestyle='--', marker='o', markersize=8, markerfacecolor='yellow')

## Ajoute des étiquettes et un titre
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Customized Plot')

## Affiche le graphique
plt.show()

## Génère des données aléatoires
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)
sizes = 1000 * np.random.rand(50)

## Crée un graphique de dispersion
plt.scatter(x, y, c=colors, s=sizes, alpha=0.5)

## Ajoute des étiquettes et un titre
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot')

## Affiche le graphique
plt.show()

names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]

plt.figure(figsize=(9, 3))

plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)

plt.suptitle('Categorical Plotting')
plt.show()

r = np.linspace(0.3, 1, 30)
theta = np.linspace(0, 4*np.pi, 30)
x = r * np.sin(theta)
y = r * np.cos(theta)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3.2))

ax1.plot(x, y, 'C3', lw=3)
ax1.scatter(x, y, s=120)
ax1.set_title('Lignes au-dessus des points')

ax2.plot(x, y, 'C3', lw=3)
ax2.scatter(x, y, s=120, zorder=2.5)  ## déplacer les points au-dessus de la ligne
ax2.set_title('Points au-dessus des lignes')

plt.tight_layout()
plt.show()