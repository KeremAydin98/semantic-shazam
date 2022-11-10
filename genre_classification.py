import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from nltk import word_tokenize
import numpy as np
from sklearn.model_selection import train_test_split
from models import *
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from clusteval import clusteval
from sklearn.datasets import make_blobs

doc2vec = Doc2Vec.load("Models/d2v.model")

def get_doc2vec_vector(song_name):

    return song_vectors[song_name]

# to find the vector of a document which is not in training data
def prepare_test_data(test_data):
    test_data = word_tokenize(test_data.lower())
    v1 = doc2vec.infer_vector(test_data)
    return v1


# Reading the dataframe
df = pd.read_csv("Data/combined-data.csv")

# Embedding vectors of songs
song_vectors = doc2vec.dv

song_names = list(df["SName"])
genres = list(df["Genres"])

# Doc2vec vectors of songs
data_points = np.array(list(map(get_doc2vec_vector,song_names)))
