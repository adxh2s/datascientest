import requests
import pandas as pd

response = requests.get("http://datascientest.com/")
print("Le code de statut pour l'endpoint \"http://datascientest.com/\" est", response.status_code)
print("La requête a pu aboutir :)\n")

response = requests.get("http://datascientest.com/corrections_examens")
print("Le code de statut pour l'endpoint \"http://datascientest.com/corrections_examens\" est", response.status_code)
print("Malheureusement, la page contenant la correction de tous les examens n'existe pas :(")


# Récupération de la liste des membres de l'équipe
response = requests.get("https://dst-moduleapi.s3.eu-west-1.amazonaws.com/members")

# Récupération du contenu de la réponse sous la forme d'un dictionnaire
response = response.json()

# Récupération de la liste des membres depuis le dictionnaire
members_id = response['members']

# Création d'une liste vide
members_data = []

# Pour chaque identifiant de la liste
for ID in members_id:
    # la requête est créée en concaténant l'endpoint avec l'identifiant
    response = requests.get("https://dst-moduleapi.s3.eu-west-1.amazonaws.com/members/" + ID)
    response = response.json()
    
    # on ajoute le contenu de la réponse à la liste
    members_data.append(response)
    
# Instanciation d'un DataFrame à partir de la liste de contenus récupérés des endpoints
df = pd.DataFrame(members_data)

print(df.head())

# "Explosion" de la colonne MemberSkillsID
df = df.explode("MemberSkillsID")

# Récupération des données "Skills"
response = requests.get("https://dst-moduleapi.s3.eu-west-1.amazonaws.com/skills")
response = response.json()

skills = response['skills']

# Conversion en DataFrame
skills_df = pd.DataFrame(skills)

# Jointure entre les deux tables
df = df.merge(skills_df, how = "left", left_on = "MemberSkillsID", right_on = "SkillID")

print(skills_df)

df = df.merge(skills_df, how='left', left_on='MemberSkillsID', right_on = "SkillID")

print(df)

print("Question (j)")
print(df['SkillName'].value_counts())
print("La compétence la plus répandue dans l'équipe est Java")

print("\nQuestion (k)")
#print(df.groupby("MemberName").count())
# ou
print(df['MemberName'].value_counts())
print("La personne avec le plus de compétences différentes est Robin")

print("\nQuestion (l)")
print(df['MemberAge'].unique().mean())
print("La moyenne d'âge dans l'équipe est de 29 ans")

def extract_skills():
    # Récupération de la liste des membres de l'équipe
    response = requests.get("https://dst-moduleapi.s3.eu-west-1.amazonaws.com/members")
    response = response.json()
    members_id = response['members']

    # Récupération des compétences des membres
    members_data = []
    
    format_requete = "https://dst-moduleapi.s3.eu-west-1.amazonaws.com/members/{ID}"

    # Pour chaque identifiant de la liste
    for ID in members_id:
        # la requête est créée en insérant l'identifiant dans le format de requête
        requete = format_requete.format(ID = ID)
        response = requests.get(requete)
        response = response.json()

        # on ajoute le contenu de la réponse à la liste
        members_data.append(response)
    
    # Instanciation d'un DataFrame à partir de la liste de contenus récupérés des endpoints
    df = pd.DataFrame(members_data)
    
    # Récupération des données "Skills"
    response = requests.get("https://dst-moduleapi.s3.eu-west-1.amazonaws.com/skills")
    response = response.json()
    skills = response['skills']

    # Conversion en DataFrame
    skills_df = pd.DataFrame(skills)
    
    return df, skills_df

df, skills_df = extract_skills()

def transform_skills(df, skills_df):
    # "Explosion" des listes dans la colonne "MemberSkillsID"
    df = df.explode("MemberSkillsID")
    
    # Jointure du `DataFrame` des membres avec celui des compétences
    df = df.merge(skills_df, left_on = "MemberSkillsID", right_on = "SkillID", how = "left")
    
    # Calcul de statistiques
    stats = df.groupby("SkillName").agg({"SkillName" : "count",
                                      "MemberAge" : "mean"})

    # On renomme les colonnes
    stats = stats.rename(columns = {
        "SkillName" : "SkillMastery",
        "MemberAge" : "AverageAge"
    })
    
    return stats

# Test des fonctions
df, skills_df = extract_skills()
stats = transform_skills(df, skills_df)
print(stats)


def load_skills(stats):
    stats.to_csv("skills_kpis.csv")
    
df, skills_df = extract_skills()

stats = transform_skills(df, skills_df)

load_skills(stats)

print(pd.read_csv("skills_kpis.csv").head(10))