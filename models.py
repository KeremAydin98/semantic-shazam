import tensorflow as tf
import tensorflow_addons as tfa


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
        self.dec_units = dec_units

        # Final dense layer
        self.fl = tf.keras.layers.Dense(vocab_size)

        # Decoder recurrent structure
        self.decoder_rnn_cell = tf.keras.layers.GRU(self.dec_units)

        # A training sampler that simply reads its inputs.
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        # Create attention mechanism
        self.attention_mechanism = tfa.seq2seq.BahdanauAttention(units=self.dec_units, memory=None,
                                                                 memory_sequence_length= self.batch_size*[max_length_input])

        # Wrap attention mechanism with the fundamental rnn cell of the decoder
        self.rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell, self.attention_mechanism,
                                                     attention_layer_size=self.dec_units)

        # Define the decoder with respect to fundamental rnn cell
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fl)

    def build_initial_state(self, batch_size, encoder_state, dtype):

        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_size, dtype=dtype)
        decoder_initial_state = decoder_initial_state.clone(hidden_state=encoder_state)

        return decoder_initial_state
    def call(self, x, initial_state):

        x = self.embedding(x)
        output, _ = self.decoder(x, initial_state=initial_state,
                                 sequence_length = self.batch_size * [max_length_output-1])

        return output


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
