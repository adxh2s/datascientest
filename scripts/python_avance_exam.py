import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from IPython.display import display

# chargement des données
df = pd.read_csv('../data/heart_disease.csv', sep=';', index_col=0)
# affichage des 5 premières lignes
df.head()


df['age'] = np.random.randint(1, 91, size=len(df))
df.describe()
# affichage des patients de 37 ans
display(df.loc[df.age == 37])
# # --> les deux patients sont malades (target=1)
# # affichage du patient d'age maximum
display(df.loc[df.age == max(df.age)])
# --> le patient de 77 ans n'est pas malade

# print(df.age.describe())


# hommes sains
nb_masc_0 = df.loc[(df.sex == 'Male') & (df.target == 0)].count()
# hommes malades
nb_masc_1 = df.loc[(df.sex == 'Male') & (df.target == 1)].count()
# femmes saines
nb_fem_0 = df.loc[(df.sex == 'Female') & (df.target == 0)].count()
# femmes malades
nb_fem_1 = df.loc[(df.sex == 'Female') & (df.target == 1)].count()

print("Nombre d'hommes sains :", nb_masc_0.sex)
print("Nombre d'hommes malades :", nb_masc_1.sex)
print("Nombre de femmes saines :", nb_fem_0.sex)
print("Nombre d'hommes malades :", nb_fem_1.sex)

# hommes et femmes saines
df_sex_0 = pd.concat([df.loc[(df.sex == 'Male') & (df.target == 0)], df.loc[(df.sex == 'Female') & (df.target == 0)]], axis=0)
# hommes et femmes malades
df_sex_1 = pd.concat([df.loc[(df.sex == 'Male') & (df.target == 1)], df.loc[(df.sex == 'Female') & (df.target == 1)]], axis=0)

# moyenne d'age individus sains
mean_age_0 = df_sex_0['age'].mean()
# moyenne d'age individus malades
mean_age_1 = df_sex_1['age'].mean()

print("moyenne d'age individus sains : ", mean_age_0)
print("moyenne d'age individus malades : ", mean_age_1)

# Remplacement des valeurs textuelles par valeurs numériques
df['sex'] = df['sex'].replace('Male', 0)
df['sex'] = df['sex'].replace('Female', 1)

# Affichage
df.head()


# On selectionne les lignes comprises entre 50 et 250
mask = ((df.thalach < 50) | (df.thalach > 250) | (df.thalach.isna()))
# moyenne de thalach sur les valeurs valides
mean_thalach = int(round(df['thalach'][~mask].mean()))
print(mean_thalach)
# On remplace ensuite les valeurs aberrantes ou manquantes
df.loc[mask, 'thalach'] = mean_thalach
# afichage des valeurs uniques de thalach
print(df.thalach.unique())
# valeurs manquantes dans df
print(df.isna().sum())


# On ne garde que les targets renseignés
df = df[df['target'].isin([0, 1])]

df.isna().sum()


# affichage
print('Valeurs manquantes ca :', df['ca'].isna().sum())
print('Valeurs manquantes exang :', df['ca'].isna().sum())
# mode de la colonne ca
ca_mode = df['ca'].mode()[0]
print('ca_mode', ca_mode)
df['ca'] = df['ca'].fillna(ca_mode)
# mode de la colonne exang
exang_mode = df['exang'].mode()[0]
print('exang_mode', exang_mode)
df['exang'] = df['exang'].fillna(exang_mode)

print(df['ca'].unique())
print(df['exang'].unique())

# affichage
print('Valeurs manquantes trestbps :', df['trestbps'].isna().sum())
print('Valeurs manquantes chol :', df['chol'].isna().sum())
# mode de la colonne trestbps
trestbps_median = df['trestbps'].median()
print('trestbps_median', trestbps_median)
df['trestbps'] = df['trestbps'].fillna(trestbps_median)
# mode de la colonne chol
chol_median = df['chol'].median()
print('chol_median', chol_median)
df['chol'] = df['chol'].fillna(chol_median)

# verification des valeurs des colonnes trestbps et chol
print(df['trestbps'].unique())
print(df['chol'].unique())
# affichage des valeurs manquantes
df.isna().sum()


# on supprime la colonne cible pour X
X = df.drop('target', axis=1)
display("X : \n", X.head())
# on garde uniquement la cible pour y
y = df['target']
display("y : \n", y)


def min_max_personnalise(col):
    """ 
    fonction pour normaliser les données
    """
    Xmin, Xmax = col.min(), col.max()
    calcul = 2 * ((col - Xmin) / (Xmax - Xmin)) - 1
    print("Colonne : ", col.name, "Xmin : ", Xmin, "Xmax : ", Xmax, "Calcul : ", calcul)
    return calcul


# Application de la fonction à chaque colonne
X_norm = X.apply(min_max_personnalise)
display("X normalisé : \n", X_norm.head())  

# on sépare les données en train et test
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size = 0.3, random_state = 10)

# on crée le modèle
lr = LogisticRegression()
# on entraine le modèle
lr.fit(X_train, y_train)
# on fait les prédictions
predictions = lr.predict(X_test)
# on affiche le tableau de contingence
print("Tableau de contingence : \n", pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted']))






