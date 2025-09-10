import numpy as np
import cv2 as cv2
import mimetypes
import matplotlib.pyplot as plt
import os
import PIL.Image as Image

p = '/home/adxh2s/Projects/datascientest/data/img/street.png'
print('cwd=', os.getcwd())
print('exists=', os.path.exists(p), '-', p)
p = '/home/adxh2s/Projects/datascientest/data/img/building.png'
print('exists=', os.path.exists(p), '-', p)
print('droits :', oct(os.stat(p).st_mode)[-3:])
img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
print(img)

try:
    Image.open(p).verify()
    print("pillow_ok=True")
except Exception as e:
    print("pillow_ok=False", e)

img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
print("cv2_none=", img is None)
if img is not None:
    print("shape=", img.shape, "dtype=", img.dtype)

# image couleur
img_color = cv2.imread("/home/adxh2s/Projects/datascientest/data/img/street.png", cv2.IMREAD_COLOR) 
# Conversion
img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)  # BGR -> RGB [9]

# Affichage
plt.imshow(img_color)                            # affichage matplotlib [8]
plt.axis("off")                                # optionnel [9]
plt.show()                                     # afficher la figure [8]

print(img_color.shape)

# rotation
def rotation(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

img_rotate = rotation(img_color,90)

plt.figure(figsize = (8,5))

plt.imshow(img_rotate)
plt.xticks([])
plt.yticks([])
plt.title('rotation')
plt.show()

# flip
plt.figure(figsize = (8,5))

plt.imshow(cv2.flip(img_color,0))
plt.xticks([])
plt.yticks([])
plt.title('flip')
plt.show()

# zoom
long, larg = img_color.shape[:2]
long_z = int(0.5*long)
larg_z = int(0.5*larg)
long_deb = 0
larg_deb = 0
img_zoom = img_color[long_deb:long_deb + long_z, larg_deb:larg_deb + larg_z, :]

plt.figure(figsize = (8,5))

plt.imshow(img_zoom)
plt.xticks([])
plt.yticks([])
plt.title('zoom')
plt.show()

# filtre
filtre = cv2.blur(img_color,ksize = (3,3))

plt.figure(figsize = (8,5))

plt.imshow(filtre)
plt.xticks([])
plt.yticks([])
plt.title('blur')
plt.show()

# pourquoi les filtres
salt_value = 20

noise = np.random.randint(salt_value+1, size=(231, 231))
img_sp_noise = img_color.copy()

#---------- Poivre---#

indexe = np.where(noise == 0)

A = indexe[0]
B = indexe[1]

img_sp_noise[A,B,:] = 0

#---------- Sel---------#

indexe = np.where(noise == salt_value)

A = indexe[0]
B = indexe[1]

img_sp_noise[A,B,:] = 255

plt.figure(figsize = (8,5))

plt.imshow(img_sp_noise)
plt.xticks([])
plt.yticks([])
plt.title('Salt and peeper')
plt.show()

# correction par filtre

filtre = cv2.medianBlur(img_sp_noise,3)

plt.figure(figsize = (8,5))

plt.imshow(filtre)
plt.xticks([])
plt.yticks([])
plt.title('medianBlur')
plt.show()

# filtre gaussien
filtre = cv2.GaussianBlur(img_color, ksize = (3,3), sigmaX = 0)

plt.figure(figsize = (8,5))

plt.imshow(filtre)
plt.xticks([])
plt.yticks([])
plt.title('GaussianBlur')
plt.show()

# techniques de seuils
# img_gray = cv2.imread("home/adxh2s/Projects/datascientest/data/img/building2.png", cv2.IMREAD_GRAYSCALE)
img_gray = cv2.imread('/home/adxh2s/Projects/datascientest/data/img/building.png', cv2.IMREAD_GRAYSCALE)

plt.figure(figsize = (8,5))
plt.imshow(img_gray, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Building')
plt.show()

filtre = cv2.GaussianBlur(img_gray, ksize = (3,3), sigmaX = 0)
seuil, img_seuillage  = cv2.threshold(filtre,115,255, type = cv2.THRESH_BINARY)

plt.figure(figsize = (8,5))

plt.imshow(img_seuillage, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.title('threshold')
plt.show()

# autre seuil
img_seuillage_2 = cv2.adaptiveThreshold(filtre,255, 
                                        adaptiveMethod  = cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        thresholdType = cv2.THRESH_BINARY,
                                        blockSize = 11,
                                        C = 5)


plt.figure(figsize = (8,5))

plt.imshow(img_seuillage_2, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.title('adaptiveThreshold')
plt.show()





