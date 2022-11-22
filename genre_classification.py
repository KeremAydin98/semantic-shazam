import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from models import create_genre_classifier

"""
Preprocessing
"""

# Loading the trained Doc2Vec vectors
doc2vec = Doc2Vec.load("Models/d2v.model")


def get_doc2vec_vector(song_name):

    return song_vectors[song_name]


# Reading the dataframe
df = pd.read_csv("Data/combined-data.csv")

# Embedding vectors of songs
song_vectors = doc2vec.dv

song_names = list(df["SName"])

genres = df["Genres"]
main_genres = ["Pop", "Romântico", "Pop/Rock", "R&B", "Rap", "Electronica",
                 "Rock", "Blues",  "Jazz", "Folk", "Country",  "Heavy Metal"]
genres = genres.map(lambda x: main_genres.index(x))

# Doc2Vec vectors of songs
data_points = np.array(list(map(get_doc2vec_vector,song_names)))
genres = tf.one_hot(genres, len(main_genres))

x_train = tf.data.Dataset.from_tensor_slices(data_points)
y_train = tf.data.Dataset.from_tensor_slices(genres)

train_dataset = tf.data.Dataset.zip((x_train, y_train))

train_dataset = train_dataset.batch(8).prefetch(tf.data.AUTOTUNE)

"""
Training
"""

# Create the model for genre classification
model = create_genre_classifier(output_size = df["Genres"].nunique())


early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=2)
model.fit(train_dataset, epochs=1000, callbacks=[early_stop])

model.save("Models/genre_classifier_model.h5")

"""
Evaluation
"""

y_pred = model.predict(data_points)

y_pred = tf.argmax(y_pred, -1)

# Actual (Left) - Predicted (Top)
confusionMatrix = confusion_matrix(genres, y_pred, labels=[0,1,2,3,4,5,6,7,8,9,10,11])

df_confusionMatrix = pd.DataFrame(data=confusionMatrix,
                                  index=["Pop", "Romântico", "Pop/Rock", "R&B", "Rap", "Electronica",
                                         "Rock", "Blues",  "Jazz", "Folk", "Country",  "Heavy Metal"],
                                  columns=["Pop", "Romântico", "Pop/Rock", "R&B", "Rap", "Electronica",
                                           "Rock", "Blues",  "Jazz", "Folk", "Country",  "Heavy Metal"])


plt.figure(figsize=(18,12))
sns.heatmap(df_confusionMatrix, cmap = 'Blues', annot = True).set_title('Confusion matrix', fontsize = 14)
plt.show()