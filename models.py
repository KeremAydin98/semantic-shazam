import tensorflow as tf
import numpy as np
import os


class Seq2SeqSummarizer:

    def __init__(self, x_voc, y_voc, embedding_dim, n_units, max_text_len):

        self.x_voc = x_voc
        self.y_voc = y_voc
        self.embedding_dim = embedding_dim
        self.n_units = n_units
        self.max_text_len = max_text_len

        """
        Encoder
        """
        encoder_inputs = tf.keras.layers.Input(shape=(self.max_text_len,))

        enc_emb = tf.keras.layers.Embedding(self.x_voc, self.embedding_dim)(encoder_inputs)

        x, hidden_state = tf.keras.layers.GRU(self.n_units,
                                              return_sequences=True,
                                              return_state=True,
                                              dropout=0.4,
                                              recurrent_dropout=0.4)(enc_emb)

        x, hidden_state = tf.keras.layers.GRU(self.n_units,
                                              return_sequences=True,
                                              return_state=True,
                                              dropout=0.4,
                                              recurrent_dropout=0.4)(x)

        encoder_outputs, hidden_state = tf.keras.layers.GRU(self.n_units,
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
        dec_emb_layer = tf.keras.layers.Embedding(self.y_voc, self.embedding_dim)
        dec_emb = dec_emb_layer(decoder_inputs)

        # Decoder GRU
        decoder_gru = tf.keras.layers.GRU(self.n_units,
                                          return_sequences=True,
                                          return_state=True,
                                          dropout=0.4,
                                          recurrent_dropout=0.4)

        x, hidden_state = decoder_gru(dec_emb, initial_state=[hidden_state])

        decoder_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.y_voc, activation="softmax"))

        decoder_outputs = decoder_dense(x)

        self.model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        self.model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam())

        """
        Inference models
        """

        # Encode the input sequence to get the feature vector
        self.encoder_model = tf.keras.models.Model(inputs=encoder_inputs,
                                                   outputs=[encoder_outputs, hidden_state])

        # Decoder setup

        # Brlow tensors will hold the states of the previous time step
        decoder_state_input_h = tf.keras.layers.Input(shape=(self.n_units,))
        decoder_hidden_state_input = tf.keras.layers.Input(shape=(self.max_text_len, self.n_units))

        # Get the embeddings of the decoder sequence
        dec_emb2 = dec_emb_layer(decoder_inputs)

        # To predict the next word in the sequence, set the initial states to the states from
        # the previous time steps
        (decoder_outputs2, state_h2) = decoder_gru(dec_emb2, initial_state=[decoder_state_input_h])

        # A dense softmax layer to generate prob dist over target vocabulary
        decoder_outputs2 = decoder_dense(decoder_outputs2)

        # Final decoder model
        self.decoder_model = tf.keras.models.Model([decoder_inputs] + [decoder_hidden_state_input,
                                                   decoder_state_input_h], [decoder_outputs2] + [state_h2])

    def load_weights(self, weights_path):

        if os.path_exists(weights_path):

            self.model.load_weights(weights_path)

    def summarize(self,input_seq, target_word_index, reverse_target_word_index, max_summary_len):

        # Encode the input as state vectors.
        (e_out, e_h, e_c) = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1
        target_seq = np.zeros((1, 1))

        # Populate the first word of target sequence with the start word.
        target_seq[0, 0] = target_word_index['START']

        stop_condition = False
        decoded_sentence = ''

        while not stop_condition:
            (output_tokens, h) = self.decoder_model.predict([target_seq]
                                                          + [e_out, e_h])

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_word_index[sampled_token_index]

            if sampled_token != 'END':
                decoded_sentence += ' ' + sampled_token

            # Exit condition: either hit max length or find the stop word.
            if sampled_token == 'END' or len(decoded_sentence.split()) >= max_summary_len - 1:
                stop_condition = True

            # Update the target sequence (of length 1)
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update internal states
            (e_h) = (h)

        return decoded_sentence


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
