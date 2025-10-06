import numpy as np
from sklearn.linear_model import LinearRegression

# Importations pour la construiction du modèle
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# Importation pour la transformation sur les images
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import Resizing

# Importation de l'utilitaire image_dataset_from_directory de Keras
from keras.utils import image_dataset_from_directory

# Importation pour les visualisations
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Définition des données d'entrée
inputs = Input(shape=(256, 256, 3))

resizing_layer = Resizing(100, 100)(inputs) 
rescaling_layer = Rescaling(1./255)(resizing_layer)  

# Construction du modèle
x = Conv2D(filters=16, kernel_size=(5, 5), padding='valid')(rescaling_layer)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)

x = Conv2D(filters=32, kernel_size=(3, 3), padding='valid')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(filters=64, kernel_size=(3, 3), padding='valid')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)

x = Flatten()(x)

x = Dense(units=64, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(units=5, activation='softmax')(x)

# Définir le modèle
model = Model(inputs=resizing_layer, outputs=outputs)

# Compilation du modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

