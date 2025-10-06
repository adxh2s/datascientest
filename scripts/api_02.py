import requests
import pandas as pd

data = requests.get("https://dst-moduleapi.s3.eu-west-1.amazonaws.com/global_transactions").json()

transactions = data['Transactions']

print("Les clés d'une transaction sont :\n", *(" "+ key + "\n" for key in transactions[0].keys()))

df_transactions = pd.DataFrame(transactions)
df_transactions.head()

print(df_transactions.dtypes)

columns_int64=['Quantity','ProductDiscount']
columns_float64=['Price']

df_transactions[columns_int64] = df_transactions[columns_int64].astype('int64')
df_transactions[columns_float64] = df_transactions[columns_float64].astype('float64')

print(df_transactions.info())

print("Valeurs manquantes:\n")
print(df_transactions.isna().sum(), "\n")

print("Doublons:", df_transactions.duplicated().sum())


# (f) À partir du DataFrame df_transactions, calculer les indicateurs de performance (KPIs) globaux suivants:
#  Vous pourrez vous aider des méthodes .groupby, .sort_values, .mean et .sum des DataFrames et Series pandas.

#     Créer une colonne TotalAmountSpent contenant le montant total de la transaction (Quantité*Prix) puis calculez le chiffre d'affaires global de l'enseigne sur l'ensemble des ventes.
df_transactions['TotalAmountSpent'] = df_transactions['Quantity'] * df_transactions['Price']
print("chiffre d'affaires global:", df_transactions['TotalAmountSpent'].sum(), "€\n")

#     Quels sont les 10 items les plus vendus ?
print("Top 10 des items les plus vendus:\n")
ventes_par_item = df_transactions.groupby("ProductName")['Quantity'].sum()
top_10_item_ventes = ventes_par_item.sort_values(ascending = False)[:10]
print(top_10_item_ventes, "\n")

#     Quels sont les 10 items ayant généré le plus de chiffre d'affaires ?
print("Top 10 des items ayant généré le plus de chiffre d'affaires:\n")
ca_par_item = df_transactions.groupby("ProductName")['TotalAmountSpent'].sum()
top_10_item_ca = ca_par_item.sort_values(ascending = False)[:10]
print(top_10_item_ca, "\n")

#     Quelle est la remise moyenne appliquée ?
print("Remise appliquée moyenne:", df_transactions['ProductDiscount'].mean(), "%\n")

#     Quelle a été la plus grosse commande effectuée en termes de chiffre d'affaires?
print("La plus grosse commande en terme de chiffre d'affaires est:\n")
print(df_transactions.sort_values("TotalAmountSpent", ascending = False).iloc[0])


# (a) En utilisant la même API qu'au premier exercice "https://dst-moduleapi.s3.eu-west-1.amazonaws.com/global_transactions", 
# récupérer cette fois-ci les données contenues dans la clé "Stores" dans un DataFrame nommé df_stores.
data = requests.get("https://dst-moduleapi.s3.eu-west-1.amazonaws.com/global_transactions").json()
df_stores = pd.DataFrame(data['Stores'])
print(df_stores.head())

# (b) Quelles sont les colonnes de df_stores qui contiennent des valeurs manquantes? 
# Calculer la proportion de valeurs manquantes pour chaque colonne.
print(df_stores.isna().mean())
# Les colonnes contenant des valeurs manquantes sont : "BestSellingProduct", "Manager" et "Surface"


# (d) Définir une fonction get_transactions_from_store_id qui prend en argument l'identifiant d'un magasin (un entier de type int )
def get_transactions_from_store_id(store_id):
    # Effectue une requête GET sur l'endpoint "https://dst-moduleapi.s3.eu-west-1.amazonaws.com/transactions/store/{}" 
    # où ID est l'identifiant du magasin donné en argument. 
    # On pourra tester l'API avec l'ID 0.
    endpoint = "https://dst-moduleapi.s3.eu-west-1.amazonaws.com/transactions/store/{}"
    data = requests.get(endpoint.format(store_id)).json()
    # Transforme les données collectées en DataFrame.    
    store_transactions = pd.DataFrame(data['Transactions'])
    # Rajoute au DataFrame une colonne StoreID contenant l'identifiant du magasin.    
    store_transactions['StoreID'] = store_id
    
    return store_transactions


# (e) Utiliser la fonction get_transactions_from_store_id avec les identifiants de la colonne "StoreId" du DataFrame df_stores 
# pour récupérer les données de transactions sur l'ensemble des magasins. 
store_transactions = []
for store_id in df_stores['StoreID']:
    store_transactions.append(get_transactions_from_store_id(store_id))

# Il faudra ensuite concaténer tous les résultats obtenus dans un DataFrame nommé df_all_stores à l'aide de la fonction pd.concat(). 
# La colonne StoreID ne doit contenir aucune valeur manquante.
df_all_stores = pd.concat(store_transactions)

print(df_all_stores.head())
print(df_all_stores.info())


# (f) Effectuer une jointure à gauche entre le DataFrame df_all_stores et le DataFrame df_Stores à l'aide de la méthode merge.
df_kpis = df_all_stores.merge(df_stores, on = 'StoreID', how = 'left')

# (h) À partir du DataFrame obtenu par jointure, construire un nouveau DataFrame contenant les caractéristiques de chaque magasin ainsi que les KPIs par magasin suivants:
#     Panier moyen par transaction --> Calcul du panier moyen
average_spend = df_kpis\
                    .groupby("StoreID")\
                    ['TotalAmountSpent']\
                    .mean()
#     Chiffre d’affaires --> chiffre d'affaires par magasin
total_sales = df_kpis\
                .groupby("StoreID")\
                ['TotalAmountSpent']\
                .sum()

# Produit le plus vendu
best_selling_product = df_kpis.groupby("StoreID")\
    .apply(lambda g: g.groupby("ProductName")['Quantity'].sum().idxmax())

# Création d'un nouveau DataFrame contenant une ligne par magasin avec ses caractéristiques
df_kpis = df_kpis\
            .drop_duplicates(subset = ['StoreID'])\
            [['StoreID', 'Staff', 'StoreCity', 'Manager', 'Surface']]\
            
# Affectation des KPIs calculés au nouveau DataFrame
df_kpis['AverageSpend'] = average_spend.values
df_kpis['TotalSales'] = total_sales.values
df_kpis['BestSellingProduct'] = best_selling_product.values
 
#     Produit le plus vendu en quantité totale (somme des unités vendues)
#     Chiffre d’affaires par employé et par m² --> Calcul des ventes par employé et par m2
df_kpis['AverageSalesByEmployee'] = df_kpis['TotalSales'] / df_kpis['Staff']
df_kpis['AverageSalesBySquareMeter'] = df_kpis['TotalSales'] / df_kpis['Surface']

# Le DataFrame sera trié du magasin le plus performant au moins performant par rapport au chiffre d'affaires.
df_kpis.sort_values(by = "TotalSales", ascending = False)
print(df_kpis.head(20))
# Attention, certains magasins vont disparaître car ils n'auront aucune transaction à leur nom. Vous devriez avoir seulement 71 magasins dans votre analyse.

transactions_endpoint = "https://dst-moduleapi.s3.eu-west-1.amazonaws.com/transactions24"
store_endpoint = "https://dst-moduleapi.s3.eu-west-1.amazonaws.com/Store/{}"
client_endpoint = "https://dst-moduleapi.s3.eu-west-1.amazonaws.com/Client/{}"
product_endpoint = "https://dst-moduleapi.s3.eu-west-1.amazonaws.com/Product/{}"

#### Récupération des transactions

transactions = requests.get(transactions_endpoint).json()    
transactions = pd.DataFrame(transactions)

#### Récupération des données clients

client_ids = transactions['ClientID'].unique()

client_data = []

print("Chargement des données client")
for clientID in client_ids:
    data = requests.get(client_endpoint.format(clientID)).json()
    client_data.append(data)
    
print("Chargement terminé")
        
df_clients = pd.DataFrame(client_data)
df_clients['ClientID'] = df_clients['ClientID'].astype(int)

#### Récupération des données produits

product_ids = transactions['ProductID'].unique()
    
product_data = []

print("Chargement des données produit")
for productID in product_ids:
    data = requests.get(product_endpoint.format(productID)).json()
    product_data.append(data)

print("Chargement terminé")

df_products = pd.DataFrame(product_data)
df_products['ProductID'] = df_products['ProductID'].astype(int)
df_products['Price'] = df_products['Price'].astype(float)

#### Récupération des données magasin

store_ids = transactions['StoreID'].unique()
    
store_data = []

print("Chargement des données magasin")
for storeID in store_ids:
    data = requests.get(store_endpoint.format(storeID)).json()
    store_data.append(data)

print("Chargement terminé")
        
df_stores = pd.DataFrame(store_data)

#### Fusion des DataFrames

df = transactions\
        .merge(df_clients, on = 'ClientID', how = "left")\
        .merge(df_products, on = 'ProductID', how = "left")\
        .merge(df_stores, on = 'StoreID', how = "left")

#### Calcul des KPIs

# Nombre total de commandes
TotalTransactions = df["ClientID"].value_counts().sort_index()

# Panier Total et Panier Moyen
df['AmountSpent'] = df['Quantity'] * df['Price']
TotalAmountSpent = df.groupby("ClientID")['AmountSpent'].sum()
AverageSpend = df.groupby("ClientID")['AmountSpent'].mean()

# Magasin et produit préférés
FavouriteStore = df.groupby("ClientID").agg({"StoreCity" : lambda x: pd.Series.mode(x)[0]})
FavouriteProduct = df.groupby("ClientID").agg({"ProductName" : lambda x: pd.Series.mode(x)[0]})

#### Agrégation des résultats

# On met les IDs clients dans le même ordre que les Series
# contenant les KPIs
df_clients = df_clients.sort_values("ClientID")
df_clients = df_clients.set_index("ClientID")

df_clients['TotalTransactions'] = TotalTransactions
df_clients['TotalAmountSpent']  = TotalAmountSpent
df_clients['AverageSpend']      = AverageSpend
df_clients['FavouriteStore']    = FavouriteStore
df_clients['FavouriteProduct']  = FavouriteProduct

# Optionnel : Dernier nettoyage
df_clients['ClientIsFidelized'] = df_clients['ClientIsFidelized'].fillna(0)
df_clients = df_clients.reset_index()
print(df_clients.head(20))









