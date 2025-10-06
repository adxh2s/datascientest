import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
    # Insérez votre code ici
from IPython.display import display

import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D


nltk.download()
stop_words = stopwords.words('english')

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!]+", " ", w)
    w = re.sub(r'\b\w{0,2}\b', '', w)

    # remove stopword
    mots = word_tokenize(w.strip())
    mots = [mot for mot in mots if mot not in stop_words]
    return ' '.join(mots).strip()


# print(f"Working directory: {os.getcwd()}")
# print(f"Script location: {__file__}")
# print(f"Files in current dir: {os.listdir('.')}")

# chargement des données
print("🔄 Chargement des données...")
try:
    # chargement du dataset IMDB
    df = pd.read_csv('data/IMDB_Dataset.csv')
    # affichage des 5 premières lignes
    display(df.head())
    print("✅ Données chargées avec succès!")
    print()
    # suppression de la colonne 'sentiment'
    df = df.drop('sentiment', axis=1)
    # prétraitement des avis
    df["review"] = df["review"].apply(lambda x :preprocess_sentence(x))
    print("✅ Prétraitement terminé!")
    print()
    print("Aperçu des données prétraitées:")
    df.head()

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(df.review)

    word2idx = tokenizer.word_index
    idx2word = tokenizer.index_word
    vocab_size = tokenizer.num_words

    def sentenceToData(tokens, WINDOW_SIZE):
        window = np.concatenate((np.arange(-WINDOW_SIZE,0),np.arange(1,WINDOW_SIZE+1)))
        X,Y=([],[])
        for word_index, word in enumerate(tokens) :
            if ((word_index - WINDOW_SIZE >= 0) and (word_index + WINDOW_SIZE <= len(tokens) - 1)) :
                X.append(word)
                Y.append([tokens[word_index-i] for i in window])
        return X, Y


    WINDOW_SIZE = 5

    X, Y = ([], [])
    for review in df.review:
        for sentence in review.split("."):
            word_list = tokenizer.texts_to_sequences([sentence])[0]
            if len(word_list) >= WINDOW_SIZE:
                Y1, X1 = sentenceToData(word_list, WINDOW_SIZE//2)
                X.extend(X1)
                Y.extend(Y1)
        
    X = np.array(X).astype(int)
    y = np.array(Y).astype(int).reshape([-1,1])


    embedding_dim = 300
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, batch_size = 128, epochs=50)

    model.save("word2vec.h5") 

    input("Press Enter to continue...")
except FileNotFoundError:
    print("❌ Erreur: Le fichier 'data/AirPassengers.csv' n'a pas été trouvé.")
    print("Vérifiez que le fichier existe dans le dossier 'data'.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    sys.exit(1)
