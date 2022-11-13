from gensim.models.doc2vec import Doc2Vec
import tensorflow as tf
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
import random
import numpy as np


# to find the vector of a document which is not in training data
def prepare_test_data(test_data):

    test_data = " ".join([word.lower() if word not in stopwords.words('english') else "" for word in test_data.lower().split()])
    test_data = word_tokenize(test_data.lower())
    v1 = doc2vec.infer_vector(test_data)
    return v1


def get_doc2vec_vector(song_name):

    return song_vectors[song_name]


doc2vec = Doc2Vec.load("Models/d2v.model")
df = pd.read_csv("Data/combined-data.csv")
model = tf.keras.models.load_model("Models/genre_classifier_model.h5")

# Embedding vectors of songs
song_vectors = doc2vec.dv


index = random.choice(np.arange(0, len(df)))
print("Random song name:",df["SName"].iloc[index])
print("Artist name: ",df["Artist"].iloc[index])
random_song_vector = get_doc2vec_vector(df["SName"].iloc[index])
print(random_song_vector)
prediction = model.predict(tf.expand_dims(random_song_vector,0))
print(prediction)
main_genres = ["Pop", "Romântico", "Pop/Rock", "R&B", "Rap", "Electronica",
                 "Rock", "Blues",  "Jazz", "Folk", "Country",  "Heavy Metal"]

print("\nPredictions:")
print("1. ", main_genres[int(tf.argmax(prediction,-1))])

prediction[0][int(tf.argmax(prediction,-1))] = 0

print("2. ",main_genres[int(tf.argmax(prediction,-1))])

prediction[0][int(tf.argmax(prediction,-1))] = 0

print("3. ",main_genres[int(tf.argmax(prediction,-1))])

print("Given genre: ",df["Genres"].iloc[index])