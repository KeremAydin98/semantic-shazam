import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from nltk import word_tokenize
import numpy as np
from sklearn.model_selection import train_test_split
from models import *
from nltk.corpus import stopwords
import tensorflow as tf

doc2vec = Doc2Vec.load("Models/d2v.model")

def get_doc2vec_vector(song_name):

    return song_vectors[song_name]

# Reading the dataframe
df = pd.read_csv("Data/combined-data.csv")

# Embedding vectors of songs
song_vectors = doc2vec.dv

song_names = list(df["SName"])

genres = df["Genres"]
main_genres = ["Rock", "Pop", "Heavy Metal", "Rap", "Pop/Rock", "R&B", "Romântico", "Country", "Folk", "Jazz", "Blues"]
genres = genres.map(lambda x: main_genres.index(x))

# Doc2vec vectors of songs
data_points = np.array(list(map(get_doc2vec_vector,song_names)))

# Create the model for genre classification
model = create_genre_classifier(output_size = df["Genres"].nunique())

# Separation of train and test datasets
X_train, X_test, y_train, y_test = train_test_split(data_points, np.array(genres), test_size=0.3, random_state=42)

y_train_one_hot = tf.one_hot(y_train, len(main_genres))
y_test_one_hot = tf.one_hot(y_test, len(main_genres))

model.fit(X_train, y_train_one_hot, validation_data=(X_test, y_test_one_hot), epochs=10)

model.save("Models/genre_classifier_model.h5")