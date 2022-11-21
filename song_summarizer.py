import config
from models import *
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv("Data/combined-data.csv")
df = df.dropna()

text_count = []
summary_count = []

for sent in df["Lyric"]:
    text_count.append(len(sent.split()))

for sent in df["Summarization"]:
    summary_count.append(len(sent.split()))

max_text_len = int(np.percentile(text_count, 98))
max_summary_len = int(np.percentile(summary_count, 98))

# Extract the lyrics and summaries of lyrics
song_lyrics = np.array(df["Lyric"])
song_summaries = np.array(df["Summarization"])

# Select text and summaries which are below the maximum lengths as defined
short_text = []
short_summary = []

for i in range(len(song_lyrics)):
    if len(song_summaries[i].split()) <= max_summary_len and len(song_lyrics[i].split()) <= max_text_len:
        short_text.append(song_lyrics[i])
        short_summary.append(song_summaries[i])

post_df = pd.DataFrame({"text":short_text, "summary":short_summary})

# Now add start of the sequence "START" and end of the sequence "END" to denote start and end of the summaries. This
# shall be used to trigger the start of the summarization during the inferencing phase
post_df["summary"] = post_df["summary"].apply(lambda x: "START " + str(x) + " END")

"""
Tokenizing the text
"""

# Train-test split
x_train, x_val, y_train, y_val = train_test_split(np.array(post_df["text"]),
                                                  np.array(post_df["summary"]),
                                                  test_size=0.1,
                                                  shuffle=True)

# Tokenize the text
x_tokenizer = tf.keras.preprocessing.text.Tokenizer()
x_tokenizer.fit_on_texts(x_train)

# Convert text into integer sequences
x_train_seq = x_tokenizer.texts_to_sequences(x_train)
x_val_seq = x_tokenizer.texts_to_sequences(x_val)

# Pad zero up to maximum length
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train_seq, maxlen=max_text_len, padding="post")
x_val = tf.keras.preprocessing.sequence.pad_sequences(x_val_seq, maxlen=max_text_len, padding="post")

# Tokenize the summary
y_tokenizer = tf.keras.preprocessing.text.Tokenizer()
y_tokenizer.fit_on_texts(y_train)

# Convert summary into integer sequences
y_train_seq = y_tokenizer.texts_to_sequences(y_train)
y_val_seq = y_tokenizer.texts_to_sequences(y_val)

# Pad zero up to maximum length
y_train = tf.keras.preprocessing.sequence.pad_sequences(y_train_seq, maxlen=max_summary_len, padding="post")
y_val = tf.keras.preprocessing.sequence.pad_sequences(y_val_seq, maxlen=max_summary_len, padding="post")


# Size of vocabulary (+1 for padding token)
x_voc = len(x_tokenizer.word_index) + 1
# Size of vocabulary (+1 for padding token)
y_voc = len(y_tokenizer.word_index) + 1

# Initialize the Seq2seq model
model = create_encoder_decoder_model(x_voc, y_voc, config.embedding_dim, config.n_units, max_text_len)

early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=2)

model.fit([x_train, y_train[:,:-1]],
          y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:,1:],
          validation_data=([x_val, y_val[:,:-1]], y_val.reshape(y_val.shape[0], y_val.shape[1],1)[:,1:]),
          epochs=50,
          batch_size=32,
          callbacks=[early_stop])



