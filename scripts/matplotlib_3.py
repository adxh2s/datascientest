import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

# abscisse --> range(7)
# ordonnée --> [3, 4, 5, 6, 5, 4, 3]
# couleur --> 'blue'
# largeur --> 0.6
# plt.bar(range(7), [3, 4, 5, 6, 5, 4, 3] , color = 'blue', width = 0.6, edgecolor = 'red', linewidth = 2, hatch = 'x')
# # affichage --> plt.show()
# plt.show()

# # Données
# categories = ['A', 'B', 'C', 'D']
# valeurs = [20, 34, 30, 35]
# barres_erreur = [2, 4, 3, 5]  # Les valeurs d'incertitude

# plt.figure(figsize=(8, 6))
# plt.bar(categories, valeurs, yerr=barres_erreur, ecolor='tomato', capsize=8)
# plt.title("Exemple de barres d'erreur colorées dans un barplot")
# plt.xlabel("Catégorie")
# plt.ylabel("Valeur")
# plt.show()

df = pd.read_csv('../data/sales_data.csv')
display(df.head())
print(df.describe())
print(df.dtypes)
print(df.info())

# plt.bar(df["Order_ID"], range(len(df)), color='green', width=0.8)
# plt.show()

# plt.bar(df["Product_Category"], range(len(df)), label='Exemple 1')
# plt.bar(df["Product_Sub-Category"], range(len(df)), label='Exemple 2', bottom=df["Product_Category"])
# plt.legend()
# plt.show()


# barWidth = 0.4

# x1 = range(12)
# x2 = [r + barWidth for r in x1 ]


# df12 = df.head(12)

# plt.bar(x1, df12["Order_Quantity"], width = barWidth, label = "Produit1")
# plt.bar(x2, df12["Order_Quantity"], width = barWidth, label = "Produit2")
# plt.xticks([0,2,4,6,8,11], ['Janvier', 'Mars', 'Mai', 'Juillet', 'Septembre', 'Decembre'])
# plt.legend()
# plt.show()


plt.scatter(df["Product_Category"], df["Product_Sub-Category"], c=df["Order_Quantity"], s=30)
plt.show()


x=[1, 2, 3, 4, 10]
plt.pie(x, labels=['A', 'B', 'C', 'D', 'E'])
plt.legend()
plt.figure(figsize=(6,6))

plt.show()


plt.figure(figsize = (7, 7))

plt.pie(x = df.head(6).Sales, labels = ['Janv', 'Fev', 'Mars', 'Avril', 'Mai', 'Juin'],
           colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple'],
           explode = [0, 0, 0, 0.2, 0, 0],
           autopct = lambda x: str(round(x, 2)) + '%',
           pctdistance = 0.7, labeldistance = 1.2,
           shadow = True)
plt.legend()
plt.show()


fig = plt.figure(figsize=(10,10))

plt.subplot(221)
plt.bar(range(len(df)), df.Order_ID, label='Commande')
plt.legend()

plt.subplot(222)
plt.scatter(df.Order_ID, df.Order_Quantity, c = 'm', label="Qantité")
plt.legend()

plt.subplot(223)
plt.plot(df.Profit,'r-*', label="Profits")
plt.legend()

plt.subplot(224)
plt.hist(df.Sales, color='green', rwidth=0.8, label="Sales")
plt.legend()

plt.show()


# Boxplot, sur une figure de 7x7
plt.figure(figsize = (7,7))
plt.boxplot(df.Profit)
plt.title( 'Boxplot')
# insertion d'un sous-graphique de type histogramme
plt.axes([0.65, 0.65, 0.2, 0.15], facecolor='#ffe5c1')
plt.hist(df.Profit, color='#FFC575')
plt.xlabel('Distribution')
plt.xticks([])
plt.yticks([])
plt.show()







