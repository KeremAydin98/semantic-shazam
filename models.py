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

