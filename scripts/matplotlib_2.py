import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import display


print('toto')
# chargement des données
df = pd.read_csv('../data/heart_disease.csv')

# affichage des 5 premières lignes
display(df.head())

x, y = [0, 2, 4, 6], [1, 4, 4, 8]

plt.plot(x, y)

plt.show()

t = np.arange(0,5,0.2)
plt.plot(t, t)
plt.plot(t, t**2)
plt.plot(t, t**3)
plt.ylim([0, 50])
plt.show()

plt.plot(t, t**2, t**3)
plt.ylim([0, 50])
plt.show()

plt.plot(t)
plt.plot(t**2)
plt.plot(t**3)
plt.ylim([0, 50])
plt.show()

plt.plot(t,t,':r',t,t**2,'-g',t,t**3,'--b')
# On plot les courbes (t,t), (t,t**2) et (t,t**3) à l'intérieur de la même fonction.

plt.ylim([0,50]);
#On limite l'axe des ordonnées à [0,50]

t = np.arange(0,5,0.2)

plt.plot(t, t, 'yh')
plt.plot(t, t**2, 'g-', linewidth=5)
plt.plot(t, t**3, 'b-D')
plt.ylim([0,50])

plt.show()

x = [50, 100, 150, 200]
y = [2, 3, 7, 10]

a = [50, 100, 150, 200]
b = [2, 7, 9, 10]

plt.plot(x, y,'b-*', linewidth=0.8, label='Trajet1')
plt.plot(a, b,'g-+', linewidth=0.8, label='Trajet2')
plt.grid(True)
plt.xlabel('Vitesse')
plt.ylabel('Temps')
plt.legend()
plt.show()


df = pd.read_csv('../data/House_Rent_Dataset.csv')
display(df.head())

df.plot()
# (x='City', y=['Rent', 'Size'], style=["r--", "c-+"], title='Ventes par mois')

x, y = [0, 0, 1, 1, 0, 0.5, 1], [1, 0, 0, 1, 1, 2, 1]

plt.plot(x, y)
plt.axis([-1, 2, -1, 2])
plt.show()