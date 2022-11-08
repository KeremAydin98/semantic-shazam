import tensorflow as tf


class Encoder(tf.keras.models.Model):

    def __init__(self):

        super().__init__()

    def call(self, input):

        x = tf.keras.layers.GRU(256, return_sequences=True)(input)
        output, state = tf.keras.layers.GRU(256, return_state=True)(x)

        return state


class Decoder(tf.keras.Model):

    def __init__(self):

        super().__init__()

    def call(self, input):

        x = tf.keras.layers.GRU(256, return_sequences=True)(input)
        output, state = tf.keras.layers.GRU(256, return_state=True)(x)

        return state

def create_genre_classifier():

    model = tf.keras.Sequential([
        tf.keras.layers.GRU(512, return_sequences=True),
        tf.keras.layers.GRU(512, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(df["Genres"].nunique(), activation="softmax"))
    ])

    return model