# Importation pour les visualisations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Importations pour la construiction du modèle
import tensorflow as tf

# Importation de l'utilitaire image_dataset_from_directory de Keras
from keras.utils import image_dataset_from_directory

# Importation pour utiliser un modèle pre entrainé
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# Importation pour la transformation sur les images
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    RandomFlip,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
)
from tensorflow.keras.models import Model

artists = pd.read_csv("/workspace/data/artists.csv", index_col=0)

print(artists.head())

# Calculer l'ordre des nationalités en fonction du nombre d'occurrences
category_order = artists['nationality'].value_counts().index

plt.figure(figsize=(15,6))
sns.countplot(y=artists["nationality"], order=category_order)
# Afficher les images
plt.savefig("paint_1.png", dpi=150, bbox_inches="tight")
print("Figure écrite: paint_1.png")

# Regrouper les données par nationalité et calculer la somme des peintures pour chaque nationalité
data = artists.groupby("nationality").sum()

plt.figure(figsize=(15,6))
sns.barplot(x=data['paintings'], y=data.index, order=data['paintings'].sort_values(ascending=False).index)
plt.savefig("paint_2.png", dpi=150, bbox_inches="tight")
print("Figure écrite: paint_2.png")

# Chargement des images en train et validation
train_ds = image_dataset_from_directory(
    "/workspace/data/images_cnn/",
    seed=42,                      
    batch_size=32,      
    validation_split=0.2,
    subset="training",          # Charger les données d'entraînement       
    image_size=(224, 224)         # Redimensionnement des images pour VGG16
)

val_ds = image_dataset_from_directory(
    "/workspace/data/images_cnn/",
    seed=42,
    batch_size=32,
    validation_split=0.2,
    subset="validation",          # Charger les données de tests
    image_size=(224, 224)         # Redimensionnement des images pour VGG16
)    

# phase de préprocessing identique à celui du modele que l'on souhaite reutilisé (vgg16)
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

# Modèle VGG16
base_model = VGG16(weights='imagenet', include_top=False)

# Freezer les couches du VGG16
base_model.trainable = False

# Création du modèle avec l'API Fonctionnelle
inputs = Input(shape=(224, 224, 3))

# Application des augmentations
x = RandomRotation(0.1)(inputs)                          
x = RandomTranslation(height_factor=0.1, width_factor=0.1)(x) 
x = RandomZoom(0.1)(x)  
x = RandomFlip("horizontal")(x)

# Construction du modèle
x = base_model(x)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(rate=0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(rate=0.2)(x)
outputs = Dense(5, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)



# Modèle VGG16
base_model = VGG16(weights='imagenet', include_top=False)

# Freezer les couches du VGG16
base_model.trainable = False

# Création du modèle avec l'API Fonctionnelle
inputs = Input(shape=(224, 224, 3))

# Application des augmentations
x = RandomRotation(0.1)(inputs)                          
x = RandomTranslation(height_factor=0.1, width_factor=0.1)(x) 
x = RandomZoom(0.1)(x)  
x = RandomFlip("horizontal")(x)

# Construction du modèle
x = base_model(x)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(rate=0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(rate=0.2)(x)
outputs = Dense(5, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_model = model.fit(train_ds, 
                          epochs=5,
                          validation_data=val_ds)

plt.figure(figsize=(12,4))

plt.subplot(121)
plt.plot(history_model.history['loss'])
plt.plot(history_model.history['val_loss'])
plt.title('Model loss by epoch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='right')

plt.subplot(122)
plt.plot(history_model.history['accuracy'])
plt.plot(history_model.history['val_accuracy'])
plt.title('Model accuracy by epoch')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='right')
# Afficher les images
plt.savefig("vgg16_1.png", dpi=150, bbox_inches="tight")
print("Figure écrite: vgg16_1.png")

for layer in base_model.layers[-4:]:
    layer.trainable = True  

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_model = model.fit(train_ds, 
                          epochs=5,
                          validation_data=val_ds)


plt.figure(figsize=(12,4))

plt.subplot(121)
plt.plot(history_model.history['loss'])
plt.plot(history_model.history['val_loss'])
plt.title('Model loss by epoch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='right')

plt.subplot(122)
plt.plot(history_model.history['accuracy'])
plt.plot(history_model.history['val_accuracy'])
plt.title('Model accuracy by epoch')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='right')
plt.savefig("vgg16_2_unfreeze.png", dpi=150, bbox_inches="tight")
print("Figure écrite: vgg16_2_unfreeze.png")