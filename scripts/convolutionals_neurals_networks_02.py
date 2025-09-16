# Pour la manipulation de tableaux
import numpy as np

# Imports nécessaires pour construire un modèle CNN avec l'API fonctionnelle de Keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D 
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Rescaling

# Pour encoder les labels
from tensorflow.keras.utils import to_categorical

# Pour évaluer les performances 
from sklearn import metrics

# Pour visualiser les performances
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

# Pour importer le datasets mnist de Keras
from tensorflow.keras.datasets.mnist import load_data

# Chargement des données MNIST
(X_train, y_train), (X_test, y_test) = load_data()

# Shape of X_train and y_train
print('Shape of X:', X_train.shape)
print('Shape of y:', y_train.shape)

# Transformez les données X_train en un tableau à 4 dimensions (nb_images, largeur, hauteur, profondeur)
# Chacune des images sera ainsi redimensionnée au format (28, 28, 1). Faire de même pour les données X_test.
X_train = X_train.reshape((-1, 28, 28, 1))
X_test = X_test.reshape((-1, 28, 28, 1))

# Transformez les labels de y_train et y_test en vecteurs catégorielles binaires (one hot), 
# grâce à la fonction to_categorical du sous-module utils de keras
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Construction de l'algorithme
# Architecture du modèle
inputs = Input(shape=(28, 28, 1))

normalization_layer = Rescaling(1./255)

# Première couche de convolution
conv_1 = Conv2D(
    filters=30,                    # Nombre de filtres
    kernel_size=(5, 5),            # Dimensions du noyau
    padding='valid',               # Mode de Dépassement
    activation='relu',             # Fonction d'activation
)

# Première couche de pooling
max_pool_1 = MaxPooling2D(pool_size=(2, 2),)

# Deuxième couche de convolution
conv_2 = Conv2D(
    filters=16,                    
    kernel_size=(3, 3),          
    padding='valid',             
    activation='relu',
)

# Deuxième couche de pooling
max_pool_2 = MaxPooling2D(pool_size=(2, 2),)

# Couche de dropout
dropout = Dropout(0.2)

# Couche de Flatten
flatten = Flatten()

# Première couche dense
dense_1 = Dense(
    units=128,
    activation='relu',
)

# Couche de sortie
dense_2 = Dense(
    units=10,
    activation='softmax',
)


# Extraction des caractéristqiues
x = normalization_layer(inputs)
x = conv_1(x)
x = max_pool_1(x)
x = conv_2(x)
x = max_pool_2(x)
x = dropout(x)

# Applatissement 
x = flatten(x)

# Couches dense pour la prédiction 
x = dense_1(x)
outputs = dense_2(x)

lenet = Model(inputs=inputs, outputs=outputs)

# compilation
lenet.compile(loss='categorical_crossentropy',  # fonction de perte
              optimizer='adam',                 # algorithme de descente de gradient
              metrics=['accuracy'])             # métrique d'évaluation

# entrainement du modele
training_history_lenet = lenet.fit(X_train, y_train,           # données
                                   validation_split=0.2,       # split de test
                                   epochs=15,                  # nombre d'epochs
                                   batch_size=128)             # taille des batchs

train_acc_lenet = training_history_lenet.history['accuracy']
val_acc_lenet = training_history_lenet.history['val_accuracy']

# On va compiler et entraîner les modèles des exercices précédents. L'opération peut prendre quelques minutes.
# Réseau Dense #####################################################

# Définition de l'entrée
inputs_dense = Input(shape=(28, 28, 1))

# Couche de rescaling
normalization_layer  = Rescaling(1./255)(inputs_dense)

# Couche de Flatten
x = Flatten()(normalization_layer)

# Première couche dense
x = Dense(
    units=20, 
    activation="relu", 
    kernel_initializer='normal'
)(x)

# Couche de sortie
outputs_dense = Dense(
    units=10, 
    activation="softmax", 
    kernel_initializer='normal'
)(x)


# CNN ############################################################
inputs_cnn = Input(shape=(28, 28, 1), name="Input")

# Couche de normalisation
normalization_layer = Rescaling(1./255)(inputs_cnn)

# Première couche de convolution
cnn_1 = Conv2D(
    filters=32, 
    kernel_size=(5, 5), 
    padding='valid', 
    activation='relu',
)(normalization_layer)

# Couche de pooling
cnn_2 = MaxPooling2D(pool_size=(2, 2),)(cnn_1)

# Couche de dropout
cnn_3 = Dropout(rate=0.2,)(cnn_2)

# Couche de Flatten
cnn_4 = Flatten()(cnn_3)

# Première couche dense
cnn_5 = Dense(
    units=128, 
    activation='relu',
)(cnn_4)

# Couche de sortie
outputs_cnn = Dense(
    units=10, 
    activation='softmax',
)(cnn_5)

################################################################

# Création des modèles
model_dense = Model(inputs=inputs_dense, outputs=outputs_dense)
model_cnn = Model(inputs=inputs_cnn, outputs=outputs_cnn)

# Compilation des modèles
model_dense.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])           
model_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])            

# Entraînement des modèles
training_history_dense = model_dense.fit(X_train, y_train, validation_split=0.2, epochs=15, batch_size=128)
training_history_cnn = model_cnn.fit(X_train, y_train, validation_split=0.2, epochs=15, batch_size=128)

val_acc_dense = training_history_dense.history['val_accuracy']
val_acc_cnn = training_history_cnn.history['val_accuracy']

# affichage
plt.figure(figsize=(10,10))
# Labels des axes
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# Courbe du score de test du réseau Dense
plt.plot(np.arange(1 , 16, 1),
         val_acc_dense,
         label = 'Dense',
         color = 'blue')

# Courbe du score de test du réseau CNN
plt.plot(np.arange(1 , 16, 1),
         val_acc_cnn, 
         label = 'CNN',
         color = 'red')

# Courbe du score de test du réseau LeNet
plt.plot(np.arange(1 , 16, 1),
         val_acc_lenet, 
         label = 'LeNet',
         color = 'green')

# Affichage de la légende
plt.legend()

# Affichage de la figure
# plt.show()
# Pour docker, sans jupyter
plt.savefig("comparaison_accuracy.png", dpi=150, bbox_inches="tight")
print("Figure écrite: comparaison_accuracy.png")

# Prédictions et classes
test_pred_lenet = lenet.predict(X_test)
test_pred_dense = model_dense.predict(X_test)
test_pred_cnn = model_cnn.predict(X_test)

test_pred_lenet_class = test_pred_lenet.argmax(axis=1)
test_pred_dense_class = test_pred_dense.argmax(axis=1)
test_pred_cnn_class = test_pred_cnn.argmax(axis=1)
y_test_class = y_test.argmax(axis=1)

# rapport de classification
print(metrics.classification_report(y_test_class, test_pred_lenet_class))

# Affichage des erreurs
plt.figure(figsize=(10,10))
error_indexes = []
for i in range(len(test_pred_cnn)):
    if (test_pred_lenet_class[i] != y_test_class[i]):
        if(test_pred_dense_class[i] != y_test_class[i]):
            if(test_pred_cnn_class[i] != y_test_class[i]):
                error_indexes += [i]


j = 1
for i in np.random.choice(error_indexes, size = 3):
    img = X_test[i] 
    img = img.reshape(28, 28)
    
    plt.subplot(1, 3, j)
    j = j + 1
    plt.axis('off')
    plt.imshow(img,cmap = cm.binary, interpolation='None')
    plt.title('True Label: ' + str(y_test_class[i]) \
              + '\n' + 'Prediction: '+ str(test_pred_lenet_class[i]) \
              + '\n' + 'Confidence: '+ str(round(test_pred_lenet[i][test_pred_lenet_class[i]], 2)))

# Affichage de la figure
# plt.show()
# Pour docker, sans jupyter
plt.savefig("erreurs.png", dpi=150, bbox_inches="tight")
print("Figure écrite: erreurs.png")

input("Appuyer sur une touche pour continuer")
