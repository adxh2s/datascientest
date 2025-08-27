import pandas as pd

# Exemple de Series
# serie = pd.Series(['12345', '6789', '12a45', '00000', '987654', '54321'])
# print(serie)
# # Vérification : valeur numérique de longueur 5
# resultat = serie.str.match(r'^\d{5}$')
# print(resultat)

# Exemple de DataFrame
data = {
    'Nom': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 40, 45],
    'Ville': ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice']
}
df = pd.DataFrame(data)
print(df)

serie  = df['Nom'].str.match(r'^[A-Z][a-z]+$')
print(serie)
