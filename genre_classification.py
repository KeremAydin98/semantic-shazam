import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from nltk import word_tokenize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from models import *

doc2vec = Doc2Vec.load("Models/d2v.model")


# to find the vector of a document which is not in training data
def prepare_test_data(test_data):
    test_data = word_tokenize(test_data.lower())
    v1 = doc2vec.infer_vector(test_data)
    return v1


# Reading the dataframe
df = pd.read_csv("Data/combined-data.csv")

# Embedding vectors of songs
song_word_vectors = doc2vec.wv
song_vectors = doc2vec.dv

song_lyrics = list(df["Clean_Lyric"])
genres = list(df["Genres"])

features = list(map(prepare_test_data,song_lyrics))
labels = genres

train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                            test_size=0.2,
                                                                            random_state=42)
train_features = np.array(train_features)
test_features = np.array(test_features)

# One hot encoding labels
one_hot = OneHotEncoder()
train_labels_one_hot = one_hot.fit_transform(np.array(train_labels).reshape(-1,1)).toarray()
test_labels_one_hot = one_hot.transform(np.array(test_labels).reshape(-1,1)).toarray()

# Generate genre classifier model
model = create_genre_classifier(n_classes=df["Genres"].nunique())

model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])


print(tf.expand_dims(train_features,0).shape, train_labels_one_hot.shape, tf.expand_dims(test_features, 0).shape, test_labels_one_hot.shape)
print(type(tf.expand_dims(train_features,0)), type(train_labels), type(tf.expand_dims(test_features, 0)), type(test_labels))
model.fit(tf.expand_dims(train_features, 0), np.array(train_labels), validation_data = (tf.expand_dims(test_features, 0), np.array(test_labels)), epochs=10)

model.save("Models/genre_classification.h5")