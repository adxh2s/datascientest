import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# charger le fichier csv dans le dataframe df
df = pd.read_csv("../data/2023_nba_player_stats.csv", index_col=0)

# afficher les 5 premières lignes du dataframe
display(df.head())

# définir le thème de seaborn
sns.set_theme(style = "ticks", context = "talk", palette = "bright")

#afficher les inforations du dataframe
display(df.info())

# afficher les statistiques descriptives du dataframe
display(df.describe())

# afficher les valeurs uniques de la colonne Age
display(df['Age'].sort_values().value_counts())

# taille de la figure
# plt.figure(figsize=(20, 20))

# ---------------------------------------
# Visualiser une colonne en particulier
# ---------------------------------------

# afficher le graphique de distribution de la colonne Age
# sns.displot(df['Age'], kde=True, bins=len(df['Age'].unique()))

# afficher le graphique de distribution de la colonne Age
# sns.displot(y=df['Age'], kde=True, color="blue")

# afficher la courbe de densité de la colonne Age
# sns.kdeplot(df['Age'], fill=True, cut=0)

# fonction de répartition cumulée
# sns.displot(df['Age'], kind="ecdf")

# graphique en barres
# sns.countplot(x=df['POS'])

# graphique en barres
# sns.countplot(x=df['Age'], hue=df["POS"])

# ----------------------------------------------------
# RELATIONS ENTRE PLUSIEURS VARIABLES QUANTITATIVES #
# ----------------------------------------------------
# Relplot est en scatter par défaut (Nuage de points)
# sns.relplot(x=df["OREB"], y=df["PTS"])

# Nuage de points entre rebond offensif et total, selon le poste
# sns.scatterplot(x=df["OREB"], y=df["REB"], hue=df["POS"])

# Nuage de points
# sns.relplot(x="STL", y="AST", size="POS", hue="POS", data=df)

# Courbe
# sns.relplot(x="Age", y="Min", data=df, kind="line")

# Courbe sans intervalle de confiance (ci=None)
# sns.relplot(x="GP", y="PF", kind='line', data=df.loc[df.GP >= 60], ci=None)

# Nuage de point avec une colonne par modalité de variable qualitative
# sns.relplot(x="GP", y="PF", kind="scatter", col="POS", data=df)

# Nuage de point avec une colonne par modalité de variable qualitative et une ligne idem
# sns.relplot(x="GP", y="PF", kind="scatter", col="POS", data=df, row="Team", style="Age", height=3)

# Nuage de point avec une colonne par modalité de variable qualitative avec un col_wrap
# sns.relplot(x="GP", y="PF", kind="scatter", col="POS", data=df, col_wrap=3, height=3)

# recherche de corrélation entre variables quantitatives
# nuage et droite de regression
# sns.lmplot(x = "Min", y = "PTS", data=df,
#            scatter_kws={"color": "blue", "s": 30, "alpha": 0.5},
#            line_kws={"color": "red", "lw": 2, "linestyle": "--"})
# plt.title("nuage et regression linéaire")

# # regression locale
# sns.lmplot(x="AST", y="TOV", data=df, lowess=True)
# plt.title("nuage et regression locale")

# # regression quadratique
# sns.lmplot(x="AST", y="TOV", data=df, order=2)
# plt.title("nuage et regression quadratique")

# Fonction pairplot
#  Pour visualiser dans un même graphique, les nuages de points entre chaque paire de variables quantitatives
# ainsi que la distribution propre de chaque variable en diagonale
# sns.pairplot(data=df[['AST','STL','POS']], hue="POS", diag_kind="hist")
# sns.pairplot(data=df, y_vars=['AST', 'STL'], x_vars=['AST', 'STL'], hue="POS", diag_kind="hist")
# Placer la légende en bas à droite
# plt.legend(loc="lower right", bbox_to_anchor=(1, 0))

# sns.pairplot(data=df[['AST','STL','POS']], hue="POS", diag_kind="kde")
# plt.title("pairplot et diag -> kde")

# Fonction heatmap
# Un moyen de mesurer les relations entre chaque paire de variables quantitatives peut être de visualiser une heatmap de la matrice de corrélation
# corr_matrix = df.drop(columns=["POS", "Team"]).corr()
# fig, ax = plt.subplots(figsize = (20,20))
# sns.heatmap(corr_matrix, ax=ax, annot = True, cmap = "coolwarm")

# ----------------------------------
# Analyse de données catégorielles #
# ----------------------------------
# Les nuages de points :
# stripplot()
# swarplot()
# Les graphiques de distribution catégorielles :
# boxplot()
# violinplot()
# boxenplot()
# Les graphiques d'estimation catégorielles :
# pointplot()
# barplot()
# countplot()
# SOIT
# catplot() qui permet de tracer de multiples graphiques en modifiant le paramètre kind au sein de la fonction.
# Par défaut, le graphique affiché par un catplot() est un stripplot : un nuage de points en bandes, chaque bande représentant une modalité de la variable catégorielle.











# affichage des figures
plt.show()



plt.subplot(121)
sns.countplot(data=top5_movies, x='title, y=coun')
plt.subplot(122)
sns.countplot(data=uk_series.head(), x='title')
plt.show()

barplot


boxenplot 



barplot -- all content