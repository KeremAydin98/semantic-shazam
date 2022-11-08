import pandas as pd
import tensorflow as tf
from gensim.models.doc2vec import Doc2Vec
from nltk import word_tokenize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

doc2vec = Doc2Vec.load("Models/d2v.model")

#to find the vector of a document which is not in training data
def prepare_test_data(test_data):
    test_data = word_tokenize(test_data.lower())
    v1 = doc2vec.infer_vector(test_data)
    return v1

# Embedding vectors of songs
song_vectors = doc2vec.dv

def return_song_embeddings(song_name):

    return song_vectors[song_name]

# Reading the dataframe
df = pd.read_csv("Data/combined-data.csv")

song_names = list(df["SName"])
genres = list(df["Genres"])

features = list(map(return_song_embeddings,song_names))
labels = genres

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)
train_features = np.array(train_features)
test_features = np.array(test_features)

# One hot encoding labels
one_hot = OneHotEncoder()
train_labels_one_hot = one_hot.fit_transform(np.array(train_labels).reshape(-1,1)).toarray()
test_labels_one_hot = one_hot.transform(np.array(test_labels).reshape(-1,1)).toarray()


model = tf.keras.Sequential([
    tf.keras.layers.Dense(512,input_shape = (20,), activation="relu"),
    tf.keras.layers.Dense(512,activation="relu"),
    tf.keras.layers.Dense(df["Genres"].nunique(), activation="softmax")
])

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

model.fit(train_features, train_labels_one_hot, validation_data = (test_features, test_labels_one_hot), epochs=10)

