import os
import random
from collections import Counter

import numpy as np
np.int = int
np.bool = bool
import shap
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    MaxPooling2D,
    Rescaling,
)
from tensorflow.keras.models import Model

try:
    from tensorflow.keras.utils import image_dataset_from_directory
except Exception:
    from keras.utils import image_dataset_from_directory

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Seeds
SEED = 24
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# Paths
DATA_ROOT = "/workspace/data/FishImgDataset"
PATH_TRAIN = os.path.join(DATA_ROOT, "train")
PATH_VAL   = os.path.join(DATA_ROOT, "val")
PATH_TEST  = os.path.join(DATA_ROOT, "test")

# Dataset params
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
LABEL_MODE = "int"

# Datasets
train_ds = image_dataset_from_directory(
    PATH_TRAIN, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode=LABEL_MODE, shuffle=True, seed=SEED,
)
val_ds = image_dataset_from_directory(
    PATH_VAL, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode=LABEL_MODE, shuffle=False,
)
test_ds = image_dataset_from_directory(
    PATH_TEST, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode=LABEL_MODE, shuffle=False,
)

# Classes
class_names_train = train_ds.class_names
class_names_val   = val_ds.class_names
class_names_test  = test_ds.class_names
print(class_names_train)
print(class_names_val)
print(class_names_test)
assert class_names_train == class_names_val == class_names_test, "class_names divergents entre splits"
class_names = class_names_train
num_classes = len(class_names)
print(num_classes)

class_counts = Counter()

for _, labels in train_ds:  
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = tf.argmax(labels, axis=1)
    class_counts.update(labels.numpy())

classes, counts = zip(*class_counts.items())
plt.bar(classes, counts)
plt.xlabel('Classes')
plt.ylabel('Nombre d\'exemples')
plt.title('Distribution des classes dans train_ds')
plt.show()

# Augmentation TensorFlow sans random.py
def augment_tf(images, labels):
    # Flip horizontal aléatoire par échantillon
    flip_mask = tf.random.uniform((tf.shape(images)[0],), seed=SEED) < 0.5
    flipped = tf.image.flip_left_right(images)
    images = tf.where(tf.reshape(flip_mask, (-1, 1, 1, 1)), flipped, images)
    # Décalage discret (roll) aléatoire jusqu'à 5% de la taille
    h, w = IMG_SIZE
    max_dx = tf.cast(0.05 * w, tf.int32)
    max_dy = tf.cast(0.05 * h, tf.int32)
    dx = tf.random.uniform((tf.shape(images)[0],), minval=-max_dx, maxval=max_dx+1, dtype=tf.int32, seed=SEED)
    dy = tf.random.uniform((tf.shape(images)[0],), minval=-max_dy, maxval=max_dy+1, dtype=tf.int32, seed=SEED)
    # Appliquer tf.roll par élément de batch
    # vectoriser avec tf.map_fn en spécifiant fn_output_signature pour supprimer l’avertissement
    def roll_one(args):
        img, sx, sy = args
        return tf.roll(tf.roll(img, shift=sy, axis=0), shift=sx, axis=1)
    images = tf.map_fn(
        roll_one,
        (images, dx, dy),
        fn_output_signature=tf.float32
    )
    return images, labels

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(augment_tf, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# Sanity shape
for x_batch, y_batch in test_ds.take(1):
    print("Test batch shape:", x_batch.shape, y_batch.shape)

# Modèle
initializer = tf.keras.initializers.GlorotUniform(seed=SEED)
def lrelu(t): return tf.nn.leaky_relu(t, alpha=0.1)

inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="Input")
x = Rescaling(1./255, name="rescale_0_1")(inputs)

x = Conv2D(32, (3,3), padding="same", kernel_initializer=initializer, bias_initializer="zeros", name="Conv_1")(x)
x = BatchNormalization(name="BN_1")(x); x = Lambda(lrelu, name="LReLU_1")(x)
x = MaxPooling2D((2,2), name="Pool_1")(x); x = Dropout(0.2, name="Drop_1")(x)

x = Conv2D(64, (3,3), padding="same", kernel_initializer=initializer, bias_initializer="zeros", name="Conv_2")(x)
x = BatchNormalization(name="BN_2")(x); x = Lambda(lrelu, name="LReLU_2")(x)
x = MaxPooling2D((2,2), name="Pool_2")(x); x = Dropout(0.25, name="Drop_2")(x)

x = Conv2D(128, (3,3), padding="same", kernel_initializer=initializer, bias_initializer="zeros", name="Conv_3")(x)
x = BatchNormalization(name="BN_3")(x); x = Lambda(lrelu, name="LReLU_3")(x)
x = MaxPooling2D((2,2), name="Pool_3")(x); x = Dropout(0.3, name="Drop_3")(x)

x = Flatten(name="Flatten")(x)
x = Dense(256, kernel_initializer=initializer, bias_initializer="zeros", name="Dense_1")(x)
x = BatchNormalization(name="BN_4")(x); x = Lambda(lrelu, name="LReLU_4")(x)
x = Dropout(0.4, name="Drop_4")(x)

outputs = Dense(num_classes, activation="softmax", kernel_initializer=initializer, bias_initializer="zeros", name="Pred")(x)
model = Model(inputs=inputs, outputs=outputs, name="FishCNN_v2")

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
]

EPOCHS = 20
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)

# Evaluation
test_metrics = model.evaluate(test_ds, verbose=0)
print(f"Test metrics (loss, acc): {test_metrics}")

# Batch sanity
test_iter = iter(test_ds)
batch_images, batch_labels = next(test_iter)
batch_preds = model.predict(batch_images, verbose=0)
pred_classes = np.argmax(batch_preds, axis=-1)
print("Batch preds unique/counts:", np.unique(pred_classes, return_counts=True))
print("Batch true unique/counts:", np.unique(batch_labels.numpy(), return_counts=True))

# Rapport + matrice
def get_predictions_and_labels(dataset):
    y_true_all, y_pred_all = [], []
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_pred = np.argmax(preds, axis=-1)
        y_true_all.append(labels.numpy())
        y_pred_all.append(y_pred)
    return np.concatenate(y_true_all, 0), np.concatenate(y_pred_all, 0)

y_true, y_pred = get_predictions_and_labels(test_ds)
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes), normalize='true')
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names, vmin=0.0, vmax=1.0)
plt.title("Matrice de confusion normalisée")
plt.xlabel("Prédit")
plt.ylabel("Vrai")
plt.tight_layout()
plt.savefig("cnn05_confusion_matrix.png", dpi=150)
print("Figure écrite: cnn05_confusion_matrix.png")


# Pour avoir le nom des classes (des poissons)
# class_names = sorted(os.listdir('val'))

# Définir le nombre d'images à afficher
number_of_images = 16

# Créer une figure pour l'affichage
plt.figure(figsize=(12,12))

# Obtenir un batch d'images depuis train_ds
for images, labels in val_ds.take(2): 
    # Afficher les images du batch
    for i in range(number_of_images):
        ax = plt.subplot(4, 4, i + 1) 
        plt.imshow(images[i].numpy().astype("uint8")) 
        plt.axis("off") 
        plt.title(f"{class_names[labels[i]]}")

# Afficher les images
plt.savefig("cnn05_tab_images.png", dpi=150)
print("Figure écrite: cnn05_tab_images.png")

X = np.array([img.numpy() for img, _ in val_ds.take(2)][0]).astype(np.uint8)
y = np.array([label.numpy() for _, label in val_ds.take(2)][0])

# Sélection des images spécifiques
images = X[:4]
labels = y[:4]

def grad_cam(image, model, layer_name):
    # Récupérer la couche convolutive
    layer = model.get_layer(layer_name)
    
    # Créer un modèle qui génère les sorties de la couche convolutive et les prédictions
    grad_model = Model(inputs=model.input, outputs=[layer.output, model.output])

    # Ajout d'une dimension de batch
    image = tf.expand_dims(image, axis=0)

    # Calcul des gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        predicted_class = tf.argmax(predictions[0])  # Classe prédite
        loss = predictions[:, predicted_class]  # Perte pour la classe prédite

    # Gradients des scores par rapport aux sorties de la couche convolutive
    grads = tape.gradient(loss, conv_outputs)

    # Moyenne pondérée des gradients pour chaque canal
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Pondération des activations par les gradients calculés
    conv_outputs = conv_outputs[0]  # Supprimer la dimension batch
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalisation de la carte de chaleur
    heatmap = tf.maximum(heatmap, 0)  # Se concentrer uniquement sur les valeurs positives
    heatmap /= tf.math.reduce_max(heatmap)  # Normaliser entre 0 et 1
    heatmap = heatmap.numpy()  # Convertir en tableau numpy pour la visualisation

   # Redimensionner la carte de chaleur pour correspondre à la taille de l'image d'origine
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (image.shape[1], image.shape[2])).numpy()
    heatmap_resized = np.squeeze(heatmap_resized, axis=-1) # supprimer la dimension de taille 1 à la fin du tableau heatmap_resized

    # Colorier la carte de chaleur avec une palette (par exemple, "jet")
    heatmap_colored = plt.cm.jet(heatmap_resized)[..., :3] # Récupérer les canaux R, G, B 

    superimposed_image = heatmap_colored * 0.7 + image[0].numpy() / 255.0

    return np.clip(superimposed_image, 0, 1), predicted_class


def show_grad_cam_cnn(images, model):
    number_of_images = images.shape[0]
    conv_layers = [layer.name for layer in model.layers if isinstance(layer, Conv2D)]

    plt.figure(figsize=(16,16))

    for j, layer in enumerate(conv_layers):

        for i in range(number_of_images):

            subplot_index = i + 1 + j * number_of_images
            plt.subplot(len(conv_layers), number_of_images, subplot_index)

            # Obtenir l'image avec la carte de chaleur superposée
            grad_cam_image, predicted_class = grad_cam(images[i], model, layer)
            
            # Afficher l'image avec Grad-CAM
            plt.title(f'Grad-CAM {layer}')
            plt.imshow(grad_cam_image)
            plt.axis("off")

    plt.savefig("cnn06_grad_cam.png", dpi=150)
    print("Figure écrite: cnn06_grad_cam.png")

show_grad_cam_cnn(images, model)

# Initialiser le masker SHAP
masker = shap.maskers.Image("inpaint_telea", images[0].shape)

# Créer l'explainer SHAP
explainer = shap.Explainer(model, masker, output_names=class_names)

# Calculer les valeurs SHAP pour les images qu'on veut expliquer 
shap_values = explainer(images, max_evals=500, outputs=shap.Explanation.argsort.flip[:4])

shap.image_plot(shap_values)
# plt.savefig("cnn06_grad_cam.png", dpi=150)
print("Figure écrite: cnn06_shap.png")

def show_feature_maps(image, model):
    # Récupérer le nom des couches de convolution du modèle
    conv_layers = [layer.name for layer in model.layers if isinstance(layer, Conv2D)]
    
    # Parcourir toutes les couches de convolution
    for j, layer in enumerate(conv_layers):
        
        # Créer un nouveau modèle qui a la même entrée que le modèle d'origine mais avec comme sortie
        # la sortie de la couche de convolution spécifique
        conv_model = Model(inputs=model.input, outputs= model.get_layer(layer).output)

        # Ajouter une dimension supplémentaire à l'image pour la rendre compatible avec le batch (shape: (1, H, W, C))
        image_batch = tf.expand_dims(image, axis=0)

        # Prédire les feature maps pour l'image donnée en utilisant le modèle créé
        feature_maps = conv_model.predict(image_batch, verbose=0)

        # Squeeze pour supprimer les dimensions inutiles, résultant en un tableau de forme (H, W, N)
        feature_maps = tf.squeeze(feature_maps)

        # Initialiser une figure pour afficher les feature maps
        plt.figure(figsize=(12, 12))

        # Parcourir toutes les feature maps de la couche
        for i in range(feature_maps.shape[-1]):
            
            # Calculer le nombre de subplots nécessaire pour afficher toutes les feature maps
            nb_subplot = feature_maps.shape[-1]**(1/2)

            # Si le nombre de subplots n'est pas un entier, arrondir à l'entier supérieur
            if nb_subplot - int(nb_subplot) != 0:
                nb_subplot = int(nb_subplot) + 1
            else: 
                nb_subplot = int(nb_subplot)

            # Créer un subplot pour chaque feature map
            plt.subplot(nb_subplot, nb_subplot, i + 1)
            plt.imshow(feature_maps[..., i])  # Afficher la feature map
            plt.axis("off")  # Désactiver les axes
            plt.title(f'Output {layer} filtre {i+1}', fontsize=16 - nb_subplot - 1)  # Ajouter un titre

    # Afficher les résultats
    plt.savefig("cnn06_features_map.png", dpi=150)
    print("Figure écrite: cnn06_features_map.png")

# Exécution de la fonction pour afficher les feature maps pour l'image spécifiée
show_feature_maps(images[0], model)