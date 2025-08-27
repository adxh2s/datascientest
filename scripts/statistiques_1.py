import pandas as pd
import numpy as np
import statsmodels.api as sm

#importation librairie seaborn
import seaborn as sns

#parametre loi normale
mu = 3
sigma = 0.1
tirages = 1000

# loi normale
simu_norm = np.random.normal(loc = mu, scale = sigma, size = tirages)

# affichage histogramme
sns.histplot(simu_norm)



# moyenne de l'échantillon
print("la moyenne de l'échantillon est :", np.mean(simu_norm))

# ecart-type de l'échantillon
print("l'écart type de l'échantillon est :", np.std(simu_norm))




# boucle pour augmenter la taille de l'échantillon
for n_size in [1000, 100000, 10000000]:
    # loi normale
    simu_norm = np.random.normal(loc = mu, scale = sigma, size = n_size)
    mean_simu =  np.mean(simu_norm)
    std_simu = np.std(simu_norm)
    print("Pour une taille d'échantillon de :", n_size, "la moyenne / ecart-type : ", mean_simu, "/", std_simu)
    
# On voit que la moyenne et l'écart type se rapprochent des valeurs théoriques 3 et 0.1



# dataset
df = pd.read_csv("youtube.csv")
# taille du dataset
print(df.shape)
# description du dataset
print(df.describe())
# affichage
df.head()


# fréquences des modalités des colonnes de df
df.describe()
# types des variables
df.dtypes
# changement de la colonne publishedAt en datetime
df.publishedAt = pd.to_datetime(df.publishedAt)



# affichage des modalités uniques de categoryId
print(df.categoryId.unique())
#affichage des fréquences des modalités de categoryId
print(df.categoryId.value_counts())


# dataset filtré
df_filtre = df.loc[df.categoryId.isin([24, 10, 20, 27])].copy()
# affichage 5 lignes
df_filtre.head()
# taille du dataset filtré
print(df_filtre.shape)


# dictionnaire anciennes / nouvelles valeurs
dict_categoryId = { 24 : 'Entertainment', 10 : 'Music', 20 : 'Gaming', 27 : 'Education'}
#remplacement
df_filtre['categoryId'] = df_filtre['categoryId'].replace(dict_categoryId)
# affichage des modalités uniques de categoryId
print(df.categoryId.unique())
#affichage des fréquences des modalités de categoryId
print(df.categoryId.value_counts())



# La categorie la plus likée est Music
# La categorie la moins likée est Education
# Entertainment et Gaming sont proches et entre les deux premières catégories






# Test Khi-2 - entre 2 variables qualitatives
# on prend un alpha classique = 0,05 pour vérifier la p-value
## Les hypothèses :
print("Les hypothèses : ")
print("H0 : Il n'y a pas de dépendance entre la categorie de la vidéo et le mois de publication")
print("H1 : Il y a une dépendance entre la categorie de la vidéo et le mois de publication")
# on veut tester si la variable categoryId est indépendante de la variable mois issu de la colonne publishedAt
#on ajoute une colonne mois à df
df['mois'] = df.publishedAt.dt.month
# on crée un tableau de contingence
contingence = pd.crosstab(df.categoryId, df.mois)
# on affiche le tableau de contingence
display(contingence)

from scipy.stats import chi2_contingency

chi2, p, dof, expected = chi2_contingency(tab_contingence)
print(chi2, "\n", p, "\n", dof, "\n", expected, "\n", )

# la p-value < 0,05, H1 est vraie
print("Conclusion : La p-value ", p, " est inférieure à 5%, donc on rejette H0 et on conclut H1")




# indicateurs de positions
mean = df.likes.mean()
mediane = df.likes.median()
q1, q2, q3 = df.likes.quantile(q=[0.25, 0.5, 0.75])
seuil_sup = q3 + 1.5 * (q3 - q1)
seuil_inf = q1 - 1.5 * (q3 - q1)
min = df.likes.min()
max = df.likes.max()

print("Moyenne : ", mean)
print("Mediane = q2 : ", mediane)
print("Min : ", min)
print("Seuil inférieur : ", seuil_inf)
print("Quartiles q1, q2 q3 : ", q1, q2, q3)
print("Seuil supérieur : ", seuil_sup)
print("Max : ", max)

# indicateurs de dispersion
std = df.likes.std()
var = df.likes.var()
print("Ecart-type : ", std)
print("Variance : ", var)
print("Ecart inter-quartile : ", q3 - q1)



max_likes = df.likes.max()
print(max_likes)
display(df.loc[df.likes == max_likes])


# sélection des colonnes numériques
df_num = df.select_dtypes(include=[np.number])
# affichage des 5 premières lignes
df_num.head()



# calculer les coefficients de corrélation des différentes colonnes numériques
df_num.corr()



# importation librairie statsmodels
import statsmodels.api as sm

# verification de la normalité de la colonne view_count ( Q-Q plot)
sm.qqplot(df_num['view_count'], line='45', fit = True)
# la variable view_count n'est pas normalement distribuée
# la variable view_count est normalement distribuée



# on veut etudier la correlation entre la colonne comments_disabled (boolean) et la colonne likes (int)
# Il s'agit d'une variable qualitative et d'une variable quantitative
# On va utiliser le test ANOVA
# Deux hypothèses :
# H0 : Il n'y a pas d'effet significatif de la variable catégorielle sur la variable continue 
# H1 : Il y a un effet significatif de la variable catégorielle sur la variable continue
# Si la p-value est inférieure à 5%, on rejette H0 et on conclut H1

# Importer la librairie 
import statsmodels.api

# Réalisation du test ANOVA et affichage de résultats
result = statsmodels.formula.api.ols('freq_card_max ~ douleur_thor', data=df).fit()
# mise en forme du tableau
table = statsmodels.api.stats.anova_lm(result)
# affichage du tableau
print(table)

## La conclusion :
print("Conclusion : La p-value (PR(>F)) est inférieure à 5% donc on rejette H0 et on conclut H1")
# on conclut donc a une influence significative de la desactivation des commentaires sur le nombre de likes









