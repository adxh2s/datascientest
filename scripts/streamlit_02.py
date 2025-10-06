import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("data/train_titanic.csv")

df.head()

st.title("Projet de classification binaire Titanic")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
    # (a) Ecrire "Introduction" en haut de la première page en utilisant la commande streamlit st.write() dans le script Python.
    st.write("### Introduction")

    # (c) Afficher les 10 premières lignes du dataframe df sur l'application web Streamlit en utilisant la méthode st.dataframe().
    st.dataframe(df.head(10))

    # (d) Afficher des informations sur le dataframe sur l'application web Streamlit en utilisant la méthode st.write() de la même façon qu'un print et la méthode st.dataframe() pour un dataframe.
    st.write(df.shape)
    st.dataframe(df.describe())

    # (e) Créer une checkbox pour choisir d'afficher ou non le nombre de valeurs manquantes en utilisant la méthode st.checkbox().
    if st.checkbox("Afficher les NA") :
      st.dataframe(df.isna().sum())



# (a) Ecrire "DataVizualization" en haut de la deuxième page en utilisant la commande st.write() dans le script Python.
if page == pages[1] : 
    st.write("### DataVizualization")

    # Nous nous intéressons à la variable cible "Survived". Cette variable prend 2 modalités : 0 si l'individu n'a pas survécu et 1 si l'individu a survécu.
    # (b) Afficher dans un plot la distribution de la variable cible.
    # Remarque : Pour afficher un countplot sur Streamlit, il faut l'encadrer de fig = plt.figure() et st.pyplot(fig)
    fig = plt.figure()
    sns.countplot(x = 'Survived', data = df)
    st.pyplot(fig)


    # (c) Afficher des plots permettant de décrire les passagers du Titanic. Ajouter des titres aux plots.
    fig = plt.figure()
    sns.countplot(x = 'Sex', data = df)
    plt.title("Répartition du genre des passagers")
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x = 'Pclass', data = df)
    plt.title("Répartition des classes des passagers")
    st.pyplot(fig)

    fig = sns.displot(x = 'Age', data = df)
    plt.title("Distribution de l'âge des passagers")
    st.pyplot(fig)

    # (d) Afficher un countplot de la variable cible en fonction du genre.
    fig = plt.figure()
    sns.countplot(x = 'Survived', hue='Sex', data = df)
    st.pyplot(fig)

    # (e) Afficher un plot de la variable cible en fonction des classes.
    fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    st.pyplot(fig)

    # (f) Afficher un plot de la variable cible en fonction des âges.
    fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
    st.pyplot(fig)

    # (g) Afficher la matrice de corrélation des variables explicatives.
    fig, ax = plt.subplots()
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), ax=ax)
    st.write(fig)



# Modélisation
# Pour terminer, nous passons à l'étape de Modélisation. Nous faisons de la classification binaire pour prédire si un passager survit ou non au nauffrage du Titanic.
# Nous faisons le preprocessing du dataframe.

# (a) Ecrire "Modélisation" en haut de la troisième page en utilisant la commande st.write() dans le script Python.
if page == pages[2] : 
    st.write("### Modélisation")

    # (b) Dans le script Python streamlit_app.py, supprimer les variables non-pertinentes (PassengerID, Name, Ticket, Cabin).

    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # (c) Dans le script Python, créer une variable y contenant la variable target. Créer un dataframe X_cat contenant les variables explicatives catégorielles et un dataframe X_num contenant les variables explicatives numériques.

    y = df['Survived']
    X_cat = df[['Pclass', 'Sex',  'Embarked']]
    X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]

    # (d) Dans le script Python, remplacer les valeurs manquantes des variables catégorielles par le mode et remplacer les valeurs manquantes des variables numériques par la médiane.
    for col in X_cat.columns:
      X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
    
    for col in X_num.columns:
      X_num[col] = X_num[col].fillna(X_num[col].median())

    # (e) Dans le script Python, encoder les variables catégorielles.
    X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)

    # (f) Dans le script Python, concatener les variables explicatives encodées et sans valeurs manquantes pour obtenir un dataframe X clean.
    X = pd.concat([X_cat_scaled, X_num], axis = 1)

    # (g) Dans le script Python, séparer les données en un ensemble d'entrainement et un ensemble test en utilisant la fonction train_test_split du package model_selection de Scikit-Learn.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # (h) Dans le script Python, standardiser les valeurs numériques en utilisant la fonction StandardScaler du package Preprocessing de Scikit-Learn.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
    X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

    # (i) Dans le script Python, créer une fonction appelée prediction qui prend en argument le nom d'un classifieur et renvoie le classifieur entrainé.
    # Remarque : On peut utiliser les classifieurs LogisticRegression, SVC et RandomForestClassifier de la librairie Scikit-Learn par exemple.
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix

    def prediction(classifier):
        if classifier == 'Random Forest':
            clf = RandomForestClassifier()
        elif classifier == 'SVC':
            clf = SVC()
        elif classifier == 'Logistic Regression':
            clf = LogisticRegression()
        clf.fit(X_train, y_train)
        return clf

    # Puisque les classes ne sont pas déséquilibrées, il est intéressant de regarder l'accuracy des prédictions. 
    # Copiez le code suivant dans votre script Python. Il crée une fonction qui renvoie au choix l'accuracy ou la matrice de confusion.
    def scores(clf, choice):
        if choice == 'Accuracy':
            return clf.score(X_test, y_test)
        elif choice == 'Confusion matrix':
            return confusion_matrix(y_test, clf.predict(X_test))

    # (j) Dans le script Python, utiliser la méthode st.selectbox() pour choisir entre le classifieur RandomForest, le classifieur SVM et le classifieur LogisticRegression. Puis retourner sur l'application web Streamlit pour visualiser la "select box".
    choix = ['Random Forest', 'SVC', 'Logistic Regression']
    option = st.selectbox('Choix du modèle', choix)
    st.write('Le modèle choisi est :', option)

    clf = prediction(option)
    display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
    if display == 'Accuracy':
        st.write(scores(clf, display))
    elif display == 'Confusion matrix':
        st.dataframe(scores(clf, display))