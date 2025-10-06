# (a) Importer la librairie requests afin d'effectuer différentes requêtes sur l'API étudiée.
import requests
import pandas as pd

# (b) A l'aide d'une requête GET, récupérer la liste des noms d'utilisateur des joueurs de la catégorie GM à partir de l'endpoint "https://api.chess.com/pub/titled/{titre}". 
# Puis, dans un dictionnaire nommé grandmasters, récupérer les données contenues dans la clé "players" de la réponse et afficher le nombre total de joueurs de niveau grand maître.
# Rappel : On peut récupérer la réponse dans un dictionnaire directement à l'aide de la commande requests.get(url).json() en utilisant l'argument headers.
headers = {'User-Agent': 'ByPass'}
grandmasters = requests.get('https://api.chess.com/pub/titled/GM', headers = headers).json()
for key, value in grandmasters.items():
    print(f"{key} : {value}")
grandmasters_players = grandmasters['players']
print("La base de données contient", len(grandmasters_players), "grand maîtres.")

# (a) Définir une fonction nommée extract_player_info prenant en argument un nom d'utilisateur de joueur et qui renvoie un dictionnaire contenant ses données.
# (b) Tester la fonction sur l'utilisateur "emilanka".
def extract_player_info(username):
    headers = {'User-Agent': 'MyApp/1.0'}
    endpoint = 'https://api.chess.com/pub/player/{}'.format(username)
    user_data = requests.get(endpoint, headers = headers).json()
    return user_data

player_infos = extract_player_info('emilanka')

for key, value in player_infos.items():
    print(f"{key} : {value}")

# (c) Importer la librairie pandas et instancier un DataFrame nommé df_gms contenant les données de tous les joueurs Grand Maître. L'opération peut prendre plusieurs minutes, donc pour vos expérimentations, lancer la boucle sur une petite partie de la liste et non la liste entière.    
gm_data = []
# Pour vos expérimentations, vous pouvez remplacer grandmasters_players par grandmasters_players[0:20] pour ne traiter que les 100 premiers joueurs.:
for username in grandmasters_players[0:25]:  
    gm_data.append(extract_player_info(username))

df_gms = pd.DataFrame(gm_data)

# résumé et informations sur le DataFrame
print(df_gms.head())
print(df_gms.info())

# nettoyage des données
df_gms = df_gms.drop(["url", "is_streamer", "avatar", "@id", "verified", "location", "status"], axis=1)
print(df_gms.info())


# (f) À l'aide de la méthode unique des Series pandas, récupérer tous les endpoints distincts de la colonne "country".
country_endpoints = df_gms['country'].unique()
# (g) Instancier un dictionnaire vide.
country_dict = {}
# Ensuite, pour chaque endpoint dans le dictionnaire, effectuer une requête GET vers cet endpoint et récupérer le nom du pays. 
# On devrait obtenir un dictionnaire où chaque item est de la forme {endpoint : nom du pays}.
for endpoint in country_endpoints:
    country_name = requests.get(endpoint, headers = headers).json()['name']
    country_dict[endpoint] = country_name
# (h) Utiliser la méthode replace des Series pandas avec ce dictionnaire pour remplacer les endpoints avec les noms de pays dans la colonne "country".
# Remplacement du endpoint par le nom du pays
df_gms['country'] = df_gms['country'].replace(country_dict)

print(df_gms.head())

# (i) Transformer les variables "last_online" et "joined" en format datetime en utilisant les secondes comme unité de décomptage de la fonction pd.to_datetime.
df_gms['last_online'] = pd.to_datetime(df_gms['last_online'], unit='s')
df_gms['joined'] = pd.to_datetime(df_gms['joined'], unit='s')

print(df_gms.head())


# (j) Afficher les 5 nationalités les plus représentées sur ce site d'échecs en ligne.
print("Nations les plus représentées au niveau GM:\n")
print(df_gms['country'].value_counts().head(5))

# (k) Afficher les 5 joueurs les plus suivis.
print("\nJoueurs les plus suivis:\n")
print(df_gms[['name', 'followers']].sort_values("followers", ascending = False).head(5))

# (l) Afficher les joueurs dont le vrai nom est connu qui ont le plus d'ancienneté chez Chess.com.
print("\nJoueurs avec le plus d'ancienneté chez chess.com :\n")
print(df_gms[['name', "joined"]].dropna().sort_values("joined").head(5))

def get_player_stats(username):
    
    endpoint = 'https://api.chess.com/pub/player/{}/stats'.format(username)

    data = requests.get(endpoint, headers = headers).json()

    return data

for key, value in get_player_stats("erik").items():
    print(f"{key} : {value}")


# (c) Récupérer dans un DataFrame nommée df_gms_stats, pour chaque joueur de niveau Grand Maître:
#     Son taux de victoire, son taux de parties nulles et son taux de défaite pour les modes "Blitz" et "Rapid".
#     Son 'rating' actuel pour chacun de ces modes de jeu.

player_data = []

for username in df_gms['username']:
    
    # Initialisation du dictionnaire
    player_kpis = {}
    player_kpis['username'] = username

    # Récupération des statistiques des joueurs
    player_stats = get_player_stats(username)
    
    # Récupération des clés de la réponse
    keys = list(player_stats.keys())

    # KPIs Blitz
    if "chess_blitz" in keys:
        blitz_stats = player_stats['chess_blitz']

        # Calcul du total du nombre de parties jouées = parties gagnées + parties perdues + parties nulles
        total_games = blitz_stats['record']['win'] + blitz_stats['record']['loss'] + blitz_stats['record']['draw']

        player_kpis['BlitzWinRate'] = blitz_stats['record']['win'] / total_games
        player_kpis['BlitzLossRate'] = blitz_stats['record']['loss'] / total_games
        player_kpis['BlitzDrawRate'] = blitz_stats['record']['draw'] / total_games
        
        player_kpis['BlitzRating'] = blitz_stats['last']['rating']
   

    # KPIs Rapid
    if "chess_rapid" in keys:
        rapid_stats = player_stats['chess_rapid']

        total_games = rapid_stats['record']['win'] + rapid_stats['record']['loss'] + rapid_stats['record']['draw']

        player_kpis['RapidWinRate'] = rapid_stats['record']['win'] / total_games
        player_kpis['RapidLossRate'] = rapid_stats['record']['loss'] / total_games
        player_kpis['RapidDrawRate'] = rapid_stats['record']['draw'] / total_games
        
        player_kpis['RapidRating'] = rapid_stats['last']['rating']
    
    player_data.append(player_kpis)
    
df_gms_stats = pd.DataFrame(player_data)
print(df_gms_stats.head(5))


# (d) Effectuer une jointure entre df_gms et df_gms_stats et stocker le résultat dans un DataFrame nommé df_gms_full.
df_gms_full = df_gms.merge(df_gms_stats, on = "username")

# (e) Afficher les données du top 5 des joueurs de Blitz 
print(df_gms_full.sort_values("BlitzRating", ascending = False).head(5))

# et du top 5 des joueurs de Rapid Chess.
print(df_gms_full.sort_values("RapidRating", ascending = False).head(5))

# (f) Calculer la moyenne du rating "Blitz" par pays et afficher les 5 pays avec la meilleure moyenne.
print(df_gms_full.groupby("country")['BlitzRating'].mean().sort_values(ascending = False).head(5))

# Est-ce que cet indicateur vous semble représentatif du niveau moyen de chaque pays aux jeux d'échecs?
# Non, car le nombre de joueurs par pays n'est pas pris en compte. Un pays avec un seul joueur très fort sera en tête du classement.

# (h) Est-ce que la corrélation entre les colonnes BlitzWinRate et BlitzRating vous semble logique?
X = df_gms_full['BlitzWinRate']
Y = df_gms_full['BlitzRating']

corr_X_Y = (X * Y).mean() - X.mean() * Y.mean()
corr_X_Y = corr_X_Y / (X.std() * Y.std())

print(corr_X_Y)

print("""
Cette corrélation ne semble pas logique car on pourrait supposer qu'un taux de victoire élevé 
impliquerait un rating élevé. Cependant, la corrélation montre que ce n'est pas toujours le cas.
Il serait intéressant de comprendre pourquoi certains joueurs ont un taux de victoire élevé
mais un rating faible.
""")



