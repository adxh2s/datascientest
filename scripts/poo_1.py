class Boisson:
    """
    Classe Boisson
    Attributs : volume en ml / peremption en j
    """
    def __init__(self, a, b):
        self.volume = a
        if b > 0:
            self.peremption = b
        else:
            self.peremption = 0
    
    def prochain_jour(self):
        if self.peremption > 0:
            self.peremption -= 1
    
    def __str__(self):
        return f"Le volume de la boisson est { self.volume } ml et sa date de péremption est dans { self.peremption } jours."
    

class Jus(Boisson):
    """
    Classe Jus
    Hérite de Boisson
    Attribut péremption systématiquement à 7
    """
    def __init__(self, a, b=7):
        self.volume = a
        self.peremption = b if b <= 7 else 7
        
        

class DataCola(Boisson):
    """
    Classe DataCola
    Hérite de Boisson
    Parametre récipient
    """
    def __init__(self, c):
        if c == 'canette':
            self.volume = 330 
            self.peremption = 60
        elif c == 'bouteille':
            self.volume = 500 
            self.peremption = 30
        else:
            self.volume = 0 
            self.peremption = 0
            

class Distributeur():
    """
    Classe Distributeur
    Attributs contenu, List de Boisson et taille, nombre max de boissons contenu dans la distributeur
    """
    def __init__(self, contenu = [], taille = 0):
        self.contenu = contenu
        self.taille = taille
        # Si le taille de boissons est plus grand que la taille du distributeur, on enleve les boissons en trop
        while len(self.contenu) > self.taille:
            self.contenu.pop()


    # affichage des informations du distributeur
    def __str__(self):
        return f"Le distributeur comporte { len(self.contenu) } boissons.\nIl reste {self.taille - len(self.contenu) } emplacements dans le distributeur."


    # affichage du contenu du distributeur
    def afficher_contenu(self):
        for boisson in self.contenu:
            print(boisson)


    # Ajout d'une boisson au contenu du distributeur si la taille le permet
    def ajouter(self, boisson):
        if len(self.contenu) < self.taille:
            self.contenu.append(boisson) 
            print(f"ajouter {boisson.__class__.__name__} / taille {len(self.contenu)}")
        else:
            print(f"Le distributeur est plein !")


    # On enleve une boisson du contenu du distributeur à un index donné
    def enlever(self, i):
        if i < len(self.contenu):
            self.contenu.pop(i)
            print(f"enlever du distributeur, il contient {len(self.contenu)} boissons maintenant")
            self.afficher_contenu()
        else:
            print(f"Le distributeur est vide !")


    # vérification de la peremption des boissons
    def perimer(self):
        for index, boisson in enumerate(self.contenu):
            if boisson.peremption == 0:
                print(f"{boisson.__class__.__name__} est périmée")
                self.enlever(index)
            else:
                print(f"{boisson.__class__.__name__} n'est pas périmée")


    # enlever un jour de peremption à toutes les boissons du distributeur
    def prochain_jour(self):
        for boisson in self.contenu:
            boisson.prochain_jour()
            print(f"prochain jour pour {boisson.__class__.__name__}, reste {boisson.peremption} jours.")


# Nouvelles boissons
jus1 = Jus(1000) 
dataCola1 = DataCola('canette')
dataCola2 = DataCola('bouteille')
dataCola3 = DataCola('damejeanne')
boisson1 = Boisson(350, 14)
boisson2 = Boisson(250, 10)
boisson3 = Boisson(150, 3)

#distributeur de taille 4
distributeur1 = Distributeur([boisson1, boisson2, boisson3, jus1, dataCola1, dataCola2, dataCola3], 5)
print(distributeur1)
distributeur1.afficher_contenu()

distributeur1.prochain_jour()
distributeur1.perimer()
print(distributeur1)
distributeur1.afficher_contenu()

distributeur1.prochain_jour()
distributeur1.perimer()
print(distributeur1)
distributeur1.afficher_contenu()

distributeur1.prochain_jour()
distributeur1.perimer()
print(distributeur1)
distributeur1.afficher_contenu()

distributeur1.ajouter(DataCola('bouteille'))
print(distributeur1)
distributeur1.afficher_contenu()

distributeur1.prochain_jour()
distributeur1.perimer()
distributeur1.prochain_jour()
distributeur1.perimer()
distributeur1.prochain_jour()
distributeur1.perimer()
distributeur1.prochain_jour()
distributeur1.perimer()

print(distributeur1)
distributeur1.afficher_contenu()
