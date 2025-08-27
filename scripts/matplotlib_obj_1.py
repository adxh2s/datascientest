from mpl_toolkits.mplot3d import Axes3D
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import argrelmin

from IPython.display import display

# Importation des données
# df = pd.read_csv('../data/sales_data.csv')
# display(df.head())
# print(df.describe())
# print(df.dtypes)
# print(df.info())

# # Création d'un graphique simple
# fig = plt.figure(figsize=(10, 6))

# x = np.arange(0, 10, 0.1)
# y = np.random.randn(len(x))
# fig = plt.figure()
# ax = fig.add_subplot(111)
# l, = ax.plot(x, y)
# t = ax.set_title('random numbers')

# plt.show()


# # création du conteneur de graphiques
# fig = plt.figure(figsize=(8, 4))

# # création de 2 graphiques 1 ligne / 2 colonnes
# ax1 = fig.add_subplot(1, 2, 1)
# ax2 = fig.add_subplot(1, 2, 2)

# # Courbe dans ax1 / on récupère l'objet Line2D dans l
# (l,) = ax1.plot([0, 2, 3], [1, 3, 2])
# l.set_color("green")
# # On récupère le titre dans t
# t = ax1.set_title("Courbe")

# # histogramme dans ax2
# ax2.hist([1, 2, 2, 2, 3, 3, 4, 5, 5])
# # On récupère le titre dans t
# t = ax2.set_title("Histogramme")

# plt.show()


# # tableau de données
# x = np.arange(-10, 10, 0.1)
# y1 = x**2
# y2 = x**3
# # Container de graphiques
# fig = plt.figure()
# # On crée un graphique avec 2 axes y
# ax1 = fig.add_subplot(111)
# ax1.plot(x, y1)
# ax1.set_ylabel("x au carré")
# # On crée un second axe y
# ax2 = ax1.twinx()  # La Commande importante ! --> Partage de l'axe x
# # On trace la courbe sur le second axe y
# ax2.plot(x, y2, "r")
# ax2.set_ylabel("x au cube")

# plt.show()


# # Graphiques avec partage d'axes
# # On crée un tableau de données
# x = np.arange(11)
# # On crée un conteneur de graphiques
# fig = plt.figure()
# # On crée 3 graphiques avec partage de l'axe x
# # Le premier graphique aura 3 lignes et 1 colonne
# # On crée le premier graphique
# ax1 = fig.add_subplot(311)
# ax1.plot(x, x)
# # On crée le second graphique
# ax2 = fig.add_subplot(312, sharex=ax1)
# ax2.plot(2 * x, 2 * x)
# # On crée le troisième graphique
# ax3 = fig.add_subplot(313, sharex=ax1)
# ax3.plot(3 * x, 3 * x)

# plt.show()

# x = np.arange(0, 10, 0.1)
# fig = plt.figure()
# ax1 = fig.add_subplot(221)
# ax1.plot(x, np.sin(x))
# ax3 = fig.add_subplot(223, sharex=ax1)
# ax3.plot(2 * x, np.sin(2 * x))

# ax2 = fig.add_subplot(222)
# ax2.plot(-x, np.cos(x))
# ax4 = fig.add_subplot(224, sharex=ax2)
# ax4.plot(x, -np.cos(x))

# plt.show()


# x = np.random.randn(500)
# fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
# print(axes)
# for i in range(2):
#     for j in range(2):
#         axes[i, j].hist(x, bins=50, color="black", alpha=0.5)


# s1, s2, s3 = (
#     np.random.randn(50).cumsum(),
#     np.random.randn(50).cumsum(),
#     np.random.randn(50).cumsum(),
# )

# fig = plt.figure(figsize=(8, 6))

# ax1 = fig.add_subplot(111)

# ax1.plot(s1, color="#33CCFF", label="courbe 1")
# ax1.plot(s2, color="#FFCC99", label="courbe2")
# ax1.plot(s3, color="#FF33CC", label="courbe3")

# ax1.set_xlim([0, 21])
# ax1.set_ylim([-15, 15])
# x_ticks = np.arange(0, 21, 2)
# ax1.set_xticks(x_ticks)
# # Afficher 'j+valeur' pour les ticks pairs, invisible sinon
# labels = [f"j+{x}" if x % 2 == 0 else "" for x in x_ticks]
# ax1.set_xticklabels(labels)
# ax1.set_xlabel("Durée après le jour j")
# ax1.legend(loc="best")

# plt.show()




# fig = plt.figure( figsize = (12, 4))
# ax1 = fig.add_subplot(111)
# dates = [dt.datetime(2017,1,17) + dt.timedelta(days=i) for i in range(10)]
# values = np.random.rand(len(dates))
# ax1.plot_date(dates, values, linestyle='-')

# plt.show()


# fig = plt.figure()
# ax= fig.add_subplot(111)
# ax.text(0.1, 0.5, r"$ f(x,y) = x^2 + 3 \times \cos(y) $", fontsize=22)
# plt.show()




# fig = plt.figure( figsize = (4, 4))
# ax = fig.add_subplot(111)
# t = ax.set_title(r"$\sin(2 \pi x)e^{-x} \ et \ les \ deux \ asymptotes \pm e^{-x}$")
# t.set_fontsize(10)
# textes = [ ax.text(3, -0.5 , 'Minimums locaux') ]
# x = np.linspace(0, 2 * np.pi, num=100)  # 21 valeurs entre -1 et 1 inclus
# y = np.sin(2 * np.pi * x) * np.exp(-x)
# ax.plot(x, y, color='blue', label=r"$\sin(2 \pi x)e^{-x}$")

# # Minimums locaux
# min_indices = argrelmin(y)
# # Ajout de marques rouges (cercles) sur les points minima
# ax.plot(x[min_indices], y[min_indices], 'dc')
# # OU
# # p=[3/4 + k for k in range(4)]
# # On implémente les 4 premiers arguments des minimums de la fonction
# # ax.plot(p, [-np.exp(-k) for k in p],'Dc')
# # On trace les points de minimus locaux
# # ax.text(3, -0.5, 'Minimums locaux', color='c',)
# # On ajoute le texte 'Minimums locaux'

# x1 = x
# y1 = np.exp(-x)
# ax.plot(x1, y1, color='green', label=r"$e^{-x}$")

# x2 = x
# y2 = -np.exp(-x)
# ax.plot(x2, y2, color='red', label=r"$-e^{-x}$")

# ax.set_xlim(0,5)
# ax.set_ylim(-1,1)
# ax.legend(loc="best")

# plt.show()



# x = np.arange(-1,1,0.05)
# y = np.arange(-2, 2, 0.05)
# X,Y = np.meshgrid(x,y)
# Z = Y**2 - X**2
# plt.contour(X,Y,Z)
# plt.show()


# x = np.arange(-2, 2, 0.01)
# y = np.arange(-2, 2, 0.01)
# X,Y = np.meshgrid(x,y)
# Z = Z = X*X/9 + Y*Y/4 - 1
# cs = plt.contour(X,Y,Z)
# cs.clabel()
# plt.show()


# x = np.linspace(-1, 1, 200)
# y = np.linspace(-1, 2, 200)
# [X,Y] = np.meshgrid(x,y)
# Z = (1 - X)**2 + (Y - X**2)**2
# cp= plt.contourf(X,Y,Z,25, cmap='jet')
# plt.colorbar()
# plt.title('Fonction de Rosenbrock')
# plt.show()


# theta = np.arange(0, 2, 1/180)*np.pi 
# plt.polar(3*theta, theta/5)
# plt.polar(theta, np.cos(4*theta))
# plt.polar(theta, [1.4]*len(theta))
# plt.show()

# theta = np.arange(0, 2, 1./180.)*np.pi
# plt.polar(theta, np.abs(np.sin(5 * theta) - 2 * np.cos(theta)))
# plt.rgrids(radii=np.linspace(0.2, 3.1, 0.7))
# plt.thetagrids(labels=np.linspace(45, 360, 90))
# plt.show()

# fig = plt.figure(figsize=(20,20))
# r = np.random.rand(150) * 2
# theta = np.random.rand(150) * 2 * np.pi
# sizes = np.random.rand(150) * 200 * r**2
# ax = plt.subplot(111, projection='polar')
# c= ax.scatter(theta, r, c=theta, s=sizes, cmap=plt.cm.hsv)


fig = plt.figure( figsize = (8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Axe X')
ax.set_ylabel('Axe Y')
ax.set_zlabel('Axe Z')
ax.set_title('Graphique 3D')
t = np.linspace(0, 2 * np.pi, 100)
x = np.sin(3 * t)
y = np.cos(3 * t)
z = np.cos(t) * np.sin(t)
ax.plot(x, y, z, label='Courbe 3d')
plt.legend()
plt.show()

fig = plt.figure(figsize = (8,6))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('Axe X')
ax.set_ylabel('Axe Y')
ax.set_zlabel('Axe Z')
ax.set_title('Graphique 3D')

z = np.linspace(0, 1, 100)
x = z * np.sin(20*z)
y = z * np.cos(20*z)
ax.scatter(x, y, z, label='Nuage 3D')

plt.legend()
plt.show()




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(0, 10, 1)
y = [2,3,4,5,1,6,2,1,7,2]
z = np.zeros(10)
y = z * np.cos(20 * z)

dx, dy = np.ones(10), np.ones(10)
dz = [1,2,3,4,5,6,7,8,9,10]

ax.bar3d(x, y, z, dx, dy, dz, color='#14c989')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

u = np.linspace(0, np.pi, 30)
v = u = np.linspace(0, 2 * np.pi, 30)
x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones(30))

ax.plot_wireframe(x, y, z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


x =  np.outer(np.linspace(-2, 2, 30), np.ones(30))
y = x.T
z = np.cos(x ** 2 + y ** 2)

fig = plt.figure(figsize= (8,6))
ax = plt.axes(projection='3d')

ax.plot_surface(x, y, z, cmap=plt.cm.Spectral_r)

plt.show()