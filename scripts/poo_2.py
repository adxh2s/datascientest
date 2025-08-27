import random

class Case():
    """
    Classe Case
    Attribut occupe
    """

    # Initialisation de la case
    def __init__(self, a = ' '):
        self.occupe = a


    # Jouer 1    
    def jouer1(self):
        if self.occupe == ' ':
            self.occupe = 'X'

    # Jouer 2
    def jouer2(self):
        if self.occupe == ' ':
            self.occupe = 'O'


    # affichage des informations
    def __str__(self):
        return self.occupe


class Terrain():
    """
    Classe Terrain
    Attributs grille et tour
    """
    def __init__(self):
        self.grille = [Case(), Case(), Case(), Case(), Case(), Case(), Case(), Case(), Case()]
        self.tour = 1

    # affichage des informations
    def __str__(self):
        morpion = ""
        for i, o_case in enumerate(self.grille):
            if i in [0,1,3,4,6,7]:
                morpion  += o_case.occupe  + "|"
            else:
                morpion += o_case.occupe  + "\n"
        return morpion


    def jouer(self, i):
        # Tour impair = joueur 1, tour pair = joueur 2
        if self.tour %2 == 1:
            self.grille[i].jouer1()
        else:
            if self.tour < 9:
                self.grille[i].jouer2()
            else:
                print("Le jeu est fini")
        # On incrémente le tour
        self.tour += 1
        # On affiche le terrain
        print(self)


    def gagnant(self):
        """
        Vérifie si le joueur a gagné
        """
        # saisies possibles
        xo_check = ['X','O']
        # On vérifie les colonnes
        for i in range(3):
            if (
                (self.grille[i].occupe in xo_check and self.grille[i+3].occupe in xo_check and self.grille[i+6].occupe in xo_check) 
                and (self.grille[i].occupe == self.grille[i+3].occupe == self.grille[i+6].occupe)
            ):
                return True
        # On vérifie les lignes
        for i in range(0, 7, 3):
            if (
                (self.grille[i].occupe in xo_check  and self.grille[i+1].occupe in xo_check and self.grille[i+2].occupe in xo_check)
                and (self.grille[i].occupe == self.grille[i+1].occupe == self.grille[i+2].occupe)
            ):
                return True
        # On vérifie les diagonales
            if (
                (self.grille[0].occupe in xo_check  and self.grille[4].occupe in xo_check and self.grille[8].occupe in xo_check) 
                and (self.grille[0].occupe == self.grille[4].occupe == self.grille[8].occupe)
            ):
                return True
            if (
                (self.grille[2].occupe in xo_check  and self.grille[4].occupe in xo_check and self.grille[6].occupe in xo_check) 
                and (self.grille[2].occupe == self.grille[4].occupe == self.grille[6].occupe)
            ):
                return True
        return False


# Test aleatoire
morpion = Terrain()
choix_joueurs = list(range(0,9))
random.shuffle(choix_joueurs)
for i in choix_joueurs:
    morpion.jouer(i)
    if morpion.gagnant():
        print(f"Le joueur {morpion.tour%2 + 1} a gagné")
        break

# Test
morpion2 = Terrain()
choix_joueurs = [4,2,3,6,5]
for i in choix_joueurs:
    morpion2.jouer(i)
    if morpion2.gagnant():
        print(f"Le joueur {morpion2.tour%2 + 1} a gagné")
        break


