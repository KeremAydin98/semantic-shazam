import tensorflow as tf
import tensorflow_addons as tfa
def create_encoder_decoder_model(vocab_size, embedding_dim, n_units, max_text_len):

    """
    Encoder
    """
    encoder_inputs = tf.keras.layers.Input(shape=(max_text_len,))

    enc_emb = tf.keras.layers.Embedding(vocab_size, embedding_dim)(encoder_inputs)

    x, hidden_state = tf.keras.layers.GRU(n_units,
                                          return_sequences=True,
                                          return_state=True,
                                          dropout=0.4,
                                          recurrent_dropout=0.4)(enc_emb)

    x, hidden_state = tf.keras.layers.GRU(n_units,
                                          return_sequences=True,
                                          return_state=True,
                                          dropout=0.4,
                                          recurrent_dropout=0.4)(x)

    encoder_outputs, hidden_state = tf.keras.layers.GRU(n_units,
                                                        return_sequences=True,
                                                        return_state=True,
                                                        dropout=0.4,
                                                        recurrent_dropout=0.4)(x)

    """
    Decoder
    """

    # Set up the decoder, using encoder_states as the initial state
    decoder_inputs = tf.keras.layers.Input(shape=(None,))

    # Embedding layer
    dec_emb_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    dec_emb = dec_emb_layer(decoder_inputs)

    # Decoder GRU
    decoder_GRU = tf.keras.layers.GRU(n_units,
                                      return_sequences=True,
                                      return_state=True,
                                      dropout=0.4,
                                      recurrent_dropout=0.4)

    x, hidden_state = decoder_GRU(dec_emb, initial_state=[hidden_state])

    decoder_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation="softmax"))

    decoder_outputs = decoder_dense(x)

    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(loss=tf.keras.losses.Sparse_Categorical_Crossentropy,
                  optimizer=tf.keras.optimizers.Adam())

    return model
"""class Encoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, enc_units):

        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.embedding_size = embedding_dim

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)

        self.GRU_layer_1 = tf.keras.layers.GRU(self.enc_units,
                                             return_sequences=True,
                                             return_state=True,
                                             dropout=0.4,
                                             recurrent_dropout=0.4)

        self.GRU_layer_2 = tf.keras.layers.GRU(self.enc_units,
                                               return_sequences=True,
                                               return_state=True,
                                               dropout=0.4,
                                               recurrent_dropout=0.4)

        self.GRU_layer_3 = tf.keras.layers.GRU(self.enc_units,
                                               return_sequences=True,
                                               return_state=True,
                                               dropout=0.4,
                                               recurrent_dropout=0.4)

    def call(self, x, hidden_state):

        encoder_inputs = tf.keras.layers.Input(shape=(config.max_text_len, ))

        x = self.embedding(encoder_inputs)

        x, hidden_state = self.GRU_layer_1(x, initial_state=hidden_state)

        x, hidden_state = self.GRU_layer_2(x, initial_state=hidden_state)

        x, hidden_state = self.GRU_layer_3(x, initial_state=hidden_state)

        return x, hidden_state

    def initialize_hidden_state(self):

        return [tf.zeros((self.batch_size, self.enc_units)), tf.zeros((self.batch_size, self.enc_units))]


class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, dec_units):

        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.vocab_size = vocab_size

        self.embedding = tf.keras.layer.Embedding(self.vocab_size, self.embedding_dim)

        self.GRU_layer = tf.keras.layers.GRU(self.dec_units, return_sequences=True,
                                             return_state=True,
                                             dropout=0.4,
                                             recurrent_dropout=0.4)

    def build_initial_state(self, batch_size, encoder_state, dtype):

        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_size, dtype=dtype)
        decoder_initial_state = decoder_initial_state.clone(hidden_state=encoder_state)

        return decoder_initial_state

    def call(self, x, initial_state):

        x = self.embedding(x)
        output, _ = self.decoder(x, initial_state=initial_state,
                                 sequence_length = self.batch_size * [max_length_output-1])

        return output"""

def create_genre_classifier(output_size):

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(output_size, activation="softmax"),
    ])

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    return model
