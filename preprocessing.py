import pandas as pd
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import config
from tqdm import tqdm
from nltk.corpus import stopwords
import random
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
from collections import Counter
from heapq import nlargest


def add_summarization(doc):

  doc = doc.replace("\n",". ")
  doc = doc.replace("..",".")
  doc = doc.replace(",.",".")

  replace_abbreviations = {"'m":" am", "'ll":" will", "'em":" them",
                           "'re":" are", "'s":" is", "n't":" not" ,
                           "’m":" am", "’ll":" will", "’em":" them",
                           "’re":" are", "’s":" is", "n’t":" not" , }

  for k,v in replace_abbreviations.items():

    doc = doc.replace(k,v)

  doc = nlp(doc)

  keyword = []
  stop_words = list(STOP_WORDS)
  pos_tag = ["PROPN", "ADJ", "NOUN", "VERB"]

  for token in doc:

    if(token.text in stop_words or token.text in string.punctuation):
      continue

    if(token.pos_ in pos_tag):
      keyword.append(token.text)

  freq_word = Counter(keyword)

  try:

    max_freq = Counter(keyword).most_common(1)[0][1]
    for word in freq_word.keys():
      freq_word[word] = (freq_word[word]/max_freq)

    sent_strength = {}
    for sent in doc.sents:
      for word in sent:
        if word.text in freq_word.keys():
          if sent in sent_strength.keys():
            sent_strength[sent] += freq_word[word.text]
          else:
            sent_strength[sent] = freq_word[word.text]

    summarized_sentences = nlargest(1, sent_strength, key=sent_strength.get)

    final_sentences = [w.text for w in summarized_sentences]

    return " ".join(final_sentences)

  except:

    return None


def remove_stopwords(text):

    return " ".join([word.lower() if word not in stopwords.words('english') else "" for word in text.lower().split()])


def trim_genres(genre):

    main_genres = ["Pop", "Romântico", "Pop/Rock", "R&B", "Rap", "Electronica",
                   "Rock", "Blues", "Jazz", "Folk", "Country", "Heavy Metal"]

    genres = genre.split(";")
    genres_list = []

    for i in range(len(genres)):

        if genres[i].strip() in main_genres:

            genres_list.append(genres[i].strip())

    if len(genres_list) > 0:

        return random.choice(genres_list)

    else:

        return None


if "combined-data.csv" not in os.listdir("Data/"):

    # Load the csv data
    df_artist = pd.read_csv("Data/artists-data.csv")
    df_lyrics = pd.read_csv("Data/lyrics-data.csv")

    # Rename the linking column to connect both dataframes
    df_artist = df_artist.rename(columns={"Link":"ALink"})

    # Merge the lyrics and artists dataframes
    df = df_artist.merge(df_lyrics, on="ALink")

    # Filter to only get English songs
    df = df[df["language"] == "en"]

    # Drop the unnecessary columns
    df = df.drop(["Songs","Popularity", "ALink", "SLink"], axis=1)

    df["Genres"] = df["Genres"].astype("string")

    df = df.dropna()

    df["Genres"] = df["Genres"].map(trim_genres)

    df = df.dropna()

    df['Clean_Lyric'] = df['Lyric'].map(remove_stopwords)

    nlp = spacy.load("en_core_web_sm")

    df["Summarization"] = df["Lyric"].map(add_summarization)

    # Pickle to use the dataframe later
    df.to_csv("Data/combined-data.csv",index=False)

else:

    # Reading the dataframe if it has already been created
    df = pd.read_csv("Data/combined-data.csv")

# Extract the lyrics and names of songs
song_lyrics = list(df["Clean_Lyric"])
song_names = list(df["SName"])

# Tag every single song lyric
tagged_lyrics = [TaggedDocument(words = word_tokenize(song_lyrics[i]), tags=[str(song_names[i])]) for
                 i in range(len(song_lyrics))]

# Initialization of Doc2Vec model
model = Doc2Vec(vector_size=config.embedding_dimension,
                alpha=config.alpha,
                min_alpha=config.min_alpha,
                min_count=config.min_count,
                dm=config.dm)

# Building vocabulary in the model
model.build_vocab(tagged_lyrics)

for epoch in tqdm(range(config.EPOCH)):

    model.train(tagged_lyrics, total_examples=model.corpus_count, epochs=model.epochs)

    # Decrease the learning rate
    model.alpha -= 0.0002

    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("Models/d2v.model")
