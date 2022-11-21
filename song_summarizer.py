import config
from models import *
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv("Data/combined-data.csv")
df = df.dropna()

# Extract the lyrics and summaries of lyrics
song_lyrics = list(df["Lyrics"])
song_summaries = list(df["Summary"])

text_count = []
summary_count = []

for sent in df["Lyrics"]:
    text_count.append(len(sent.split()))

for sent in df["Summary"]:
    summary_count.append(len(sent.split()))

print(np.percentile(text_count, 95))
print(np.percentile(summary_count, 95))

# Text encoder stack
encoder = Encoder(config.vocab_size, config.embedding_dim, config.enc_units, config.batch_size)

lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')
lang_tokenizer.fit_on_texts(lang)

train_dataset, val_dataset, inp_lang, targ_lang = dataset_creator.call(config.num_examples, config.BUFFER_SIZE,
                                                                       config.BATCH_SIZE)



