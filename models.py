import tensorflow as tf


class Encoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):

        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.GRU_layer = tf.keras.layers.GRU(self.enc_units,
                                             return_sequences=True,
                                             return_state=True)

    def call(self, x, hidden_state):

        x = self.embedding(x)
        output, hidden_state = self.GRU_layer(x, initial_state=hidden_state)

        return output, hidden_state

    def initialize_hidden_state(self):

        return [tf.zeros((self.batch_size, self.enc_units)), tf.zeros((self.batch_size, self.enc_units))]


class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):

        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # Final dense layer
        self.fl = tf.keras.layers.Dense(vocab_size)


    def call(self, x):

        x = tf.keras.layers.GRU(256, return_sequences=True)(x)
        output, state = tf.keras.layers.GRU(256, return_state=True)(x)

        return state


def create_genre_classifier(output_size):

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(output_size, activation="softmax"),
    ])

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    return model
