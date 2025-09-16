import os, sys
from pathlib import Path
print(f"Python: {sys.version}")
print(f"CWD: {os.getcwd()}")
try:
    script_dir = Path(__file__).parent.resolve()
except NameError:
    script_dir = Path.cwd().resolve()
print(f"Script dir: {script_dir}")
print("CWD content:", list(Path('.').glob('*')))
print("Script dir content:", list(script_dir.glob('*')))

# Importations pour la construction du modèle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout

# Importation pour la transformation sur les images
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import Resizing
from tensorflow.keras.layers import RandomFlip
from tensorflow.keras.layers import RandomZoom
from tensorflow.keras.layers import RandomRotation
from tensorflow.keras.layers import RandomBrightness
from tensorflow.keras.layers import RandomContrast
from tensorflow.keras.layers import RandomTranslation 

# Importation de l'utilitaire image_dataset_from_directory de Keras
from keras.utils import image_dataset_from_directory

# Pour évaluer les performances 
from sklearn import metrics

# Pour visualiser les performances
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

data_dir = "/workspace/data/face_age/"  

train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,       # Fraction des données utilisée pour la validation
    subset="training",          # Charger les données d'entraînement
    seed=42,                    # Graine pour le découpage des données
    batch_size=64               # Taille des lots
)

val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,       # Fraction des données utilisée pour la validation
    subset="validation",        # Charger les données de validation
    seed=42,
    batch_size=64
)

# Nombre de lot dans l'ensemble d'entraînement
print("Nombre de batch dans train_ds:", train_ds.cardinality().numpy())

# Nombre de lot dans l'ensemble de validation
print("Nombre de batch dans val_ds:", val_ds.cardinality().numpy())


# Définir le nombre d'images à afficher
number_of_images = 6

# Créer une figure pour l'affichage
plt.figure(figsize=(10,10))

# Obtenir un batch d'images depuis train_ds
for images, labels in train_ds.take(1):
    # Afficher les images du batch
    for i in range(number_of_images):
        ax = plt.subplot(2, 3, i + 1) 
        plt.imshow(images[i].numpy().astype("uint8"))  # Convertir l'image en entier pour l'affichage
        plt.axis("off") 
        plt.title(f"Age: {labels[i].numpy() + 18}")  # Afficher le label de l'image (+18 car nos données d'age commence à 18 et nos labels à 0)

# Afficher les images
plt.savefig("img_1.png", dpi=150, bbox_inches="tight")
print("Figure écrite: comparaison_accuracy.png")

# Charger une image de train_ds
for images, labels in train_ds.take(1):  # Prendre un lot d'images et de labels
    first_image = images[0]  # Sélectionner la première image du lot
    break 

# Définir les couches de transformation
random_translation = RandomTranslation(0.2, 0.2)   # Étirement
random_zoom = RandomZoom(0.2)                      # Agrandissement
random_flip = RandomFlip("horizontal")             # Retournement horizontal

# Appliquer les transformations 
x = random_translation(first_image)  
x = random_zoom(x)                   
x = random_flip(x)              


plt.figure(figsize=(12,12))

# Afficher l'image originale
plt.subplot(121) 
plt.imshow(first_image.numpy().astype("uint8")) 
plt.title("Image originale")
plt.axis("off")

# Afficher l'image transformée
plt.subplot(122) 
plt.imshow(x.numpy().astype("uint8"))
plt.title("Image transformée")
plt.axis("off")

# Afficher les images
plt.savefig("img_2.png", dpi=150, bbox_inches="tight")
print("Figure écrite: comparaison_accuracy.png")

from tensorflow.keras.callbacks import Callback
from timeit import default_timer as timer

class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)


from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

early_stopping = EarlyStopping(
                                patience=5, # Attendre 5 epochs avant application
                                min_delta=0.01, # si au bout de 5 epochs la fonction de perte ne varie pas de 1%, 
    # que ce soit à la hausse ou à la baisse, on arrête
                                verbose=1, # Afficher à quel epoch on s'arrête
                                mode='min',
                                monitor='val_loss')

reduce_learning_rate = ReduceLROnPlateau(
                                    monitor="val_loss",
                                    patience=3, # si val_loss stagne sur 3 epochs consécutives selon la valeur min_delta
                                    min_delta=0.01,
                                    factor=0.1,  # On réduit le learning rate d'un facteur 0.1
                                    cooldown=4,  # On attend 4 epochs avant de réitérer 
                                    verbose=1)

time_callback = TimingCallback()


# Définition de l'entrée du modèle
inputs = Input(shape=(256, 256, 3))

# Transformation des images : redimensionnement, normalisation et augmentation
x = Resizing(50, 50)(inputs)    # Redimensionner les images à 100x100 pixels
x = Rescaling(1./255)(x)        # Normalisation des pixels pour avoir des valeurs entre 0 et 1
x = RandomFlip("horizontal")(x) # Retourner les images horizontalement de façon aléatoire
x = RandomRotation(0.2)(x)      # Appliquer une rotation aléatoire entre -0.2 et +0.2
x = RandomZoom(0.2)(x)          # Appliquer un zoom aléatoire entre 0.8 et 1.2
x = RandomContrast(0.2)(x)      # Modifier le contraste de l'image de façon aléatoire
x = RandomBrightness(0.1)(x)    # Appliquer une variation de la luminosité de l'image de -0.1 à +0.1

# Ajout de la couche de convolution
x = Conv2D(filters=32, 
           kernel_size=(3, 3), 
           activation="relu",
           padding="valid")(x)

# Ajout de la couche de pooling pour réduire la taille des données
x = MaxPooling2D(pool_size=(2, 2), padding='valid')(x)

# Ajout d'une couche de dropout pour éviter le surapprentissage
x = Dropout(0.3)(x)

# Applatir les données pour les passer à la couche dense
x = Flatten()(x)

# SCouche dense pour faire la prédiction
outputs = Dense(1, activation="linear")(x) 

# Définir le modèle avec les entrées et sorties spécifiées
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss="mse", metrics=['mean_absolute_error'])

model_history = model.fit(train_ds,
                          validation_data=val_ds,
                          epochs=50,
                          callbacks = [reduce_learning_rate,
                                       early_stopping,
                                       time_callback]) 

train_loss = model_history.history["loss"]
val_loss = model_history.history["val_loss"]

train_mae = model_history.history["mean_absolute_error"]
val_mae = model_history.history["val_mean_absolute_error"]


plt.figure(figsize=(20, 8))

# Tracer la perte MSE
plt.subplot(121)
plt.plot(train_loss)
plt.plot(val_loss)
plt.title('Perte du modèle par époque (MSE)')
plt.ylabel('Perte (MSE)')
plt.xlabel('Époque')
plt.legend(['Entraînement', 'Validation'], loc='best')

# Tracer l'erreur absolue moyenne (MAE)
plt.subplot(122)
plt.plot(train_mae)
plt.plot(val_mae)
plt.title('Erreur absolue moyenne par époque (MAE)')
plt.ylabel('Erreur absolue moyenne (MAE)')
plt.xlabel('Époque')
plt.legend(['Entraînement', 'Validation'], loc='best')

# Afficher les images
plt.savefig("img_3.png", dpi=150, bbox_inches="tight")
print("Figure écrite: comparaison_accuracy.png")