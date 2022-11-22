from gensim.models.doc2vec import Doc2Vec
import tensorflow as tf
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
import random
import numpy as np
import pickle
from models import Seq2SeqSummarizer
import config


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
df = df.dropna()
genre_classifier_model = tf.keras.models.load_model("Models/genre_classifier_model.h5")

seq2seq = Seq2SeqSummarizer(config.x_voc, config.y_voc, config.embedding_dim, config.n_units, config.max_text_len)

seq2seq.load_weights("Models/summarizer_weights")

# Embedding vectors of songs
song_vectors = doc2vec.dv

# loading summarizer tokenizers
with open('Pickles/x_tokenizer.pickle', 'rb') as handle:
    x_tokenizer = pickle.load(handle)
with open('Pickles/y_tokenizer.pickle', 'rb') as handle:
    y_tokenizer = pickle.load(handle)

# Tokenizer index dictionaries
reverse_target_word_index = y_tokenizer.index_word
reverse_source_word_index = x_tokenizer.index_word
target_word_index = y_tokenizer.word_index

for _ in range(5):

    index = random.choice(np.arange(0, len(df)))
    print("Random song name:",df["SName"].iloc[index])
    print("Artist name: ",df["Artist"].iloc[index])
    random_song_vector = get_doc2vec_vector(df["SName"].iloc[index])
    print("Given genre: ",df["Genres"].iloc[index],"\n")

    prediction = genre_classifier_model.predict(tf.expand_dims(random_song_vector,0))
    main_genres = ["Pop", "Romântico", "Pop/Rock", "R&B", "Rap", "Electronica",
                   "Rock", "Blues",  "Jazz", "Folk", "Country",  "Heavy Metal"]
    print("\nPredictions:")
    print("1. ", main_genres[int(tf.argmax(prediction,-1))])

    prediction[0][int(tf.argmax(prediction,-1))] = 0

    print("2. ",main_genres[int(tf.argmax(prediction,-1))])

    prediction[0][int(tf.argmax(prediction,-1))] = 0

    print("3. ",main_genres[int(tf.argmax(prediction,-1))])

    print("\n")

    seq_lyrics = x_tokenizer.texts_to_sequences(df["Lyrics"].iloc[index])

    print(f"Given summary:\n{df['Summarization'].iloc[index]}")
    print(f"Predicted summary:\n{seq2seq.summarize(seq_lyrics, target_word_index, reverse_target_word_index, config.max_summary_len)}")

    print("\n")

    print(f"Most similar songs:\n{doc2vec.most_similar([random_song_vector])}")

    print("--------------------------------------------\n")


