
import numpy as np
import cv2 #import OpenCV
import matplotlib.pyplot as plt
import os

p = '/home/adxh2s/Projects/datascientest/data/img/street.png'
print('cwd=', os.getcwd())
print('exists=', os.path.exists(p))

# Charger une image depuis un fichier
# image couleur
img_color = cv2.imread("/home/adxh2s/Projects/datascientest/data/img/street.png", cv2.IMREAD_COLOR) 

# Conversion BGR --> RGB obligatoire pour affichage
img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)  # BGR -> RGB [9]

# image en niveaux de gris
img_gray = cv2.imread("/home/adxh2s/Projects/datascientest/data/img/street.png", cv2.IMREAD_GRAYSCALE) 
# Passage par OpenCV directement pour afficher l'image
#cv2.imshow("Image", img_color)       # ouvre une fenêtre GUI [1]
#cv2.imshow("Image Grey", img_gray)       # ouvre une fenêtre GUI [1]

# Type des images ((> Array))
print('le type de img_gray est :', type(img_gray))
print('le type de img_color est :', type(img_color))

# Taille des images
print(img_color.shape)
print(img_gray.shape)


# Affichage
plt.imshow(img_color)                            # affichage matplotlib [8]
plt.axis("off")                                # optionnel [9]
plt.show()                                     # afficher la figure [8]

# Affichage
plt.imshow(img_gray, cmap= 'gray')             # affichage matplotlib [8]
plt.axis("off")                                # optionnel [9]
plt.show()                                     # afficher la figure [8]

# On récupère la taille de l'image en hauteur/largeur
height, width = img_gray.shape

# On met les pixels en haut à gauche à 0 (noir)
for h in range(round(height / 2)):
    for w in range(round(width / 2)):
        img_gray[h, w] = 0
        
plt.figure(figsize = (8,5))

plt.imshow(img_gray, cmap = 'gray')
#plt.xticks([])
#plt.yticks([])

plt.show()

# On détermine les sommets de notre polygone --> on veut laisser un triangle de visualisation
sommets = [
    (0, height),
    (width / 2, height / 2),
    (width, height),
]

sommets =  np.array([sommets], np.int32)

# mask avec un tableau de 0 de la taille de l'image
mask = np.zeros_like(img_gray)    
      
# On superpose le polygone sur le masque
mask = cv2.fillPoly(mask, sommets, color = 255)
# On additionne les deux images en commençant par l'image de base puis le masque
masked_image = cv2.bitwise_and(img_gray, mask)
# On affiche
plt.figure(figsize = (8,5))
plt.imshow(masked_image, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.show()

fig = plt.figure(figsize = (12,12))

fig.add_subplot(1,2,1)
plt.imshow(cv2.resize(img_color, dsize = (150,150),interpolation = cv2.INTER_AREA ))
plt.xticks([])
plt.yticks([])

fig.add_subplot(1,2,2)
plt.imshow(cv2.resize(img_color, dsize = (1000,1000)))
plt.xticks([])
plt.yticks([])

plt.show()