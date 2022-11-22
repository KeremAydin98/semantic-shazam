import tensorflow as tf


def create_encoder_decoder_model(x_voc, y_voc, embedding_dim, n_units, max_text_len):

    """
    Encoder
    """
    encoder_inputs = tf.keras.layers.Input(shape=(max_text_len,))

    enc_emb = tf.keras.layers.Embedding(x_voc, embedding_dim)(encoder_inputs)

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
    dec_emb_layer = tf.keras.layers.Embedding(y_voc, embedding_dim)
    dec_emb = dec_emb_layer(decoder_inputs)

    # Decoder GRU
    decoder_gru = tf.keras.layers.GRU(n_units,
                                      return_sequences=True,
                                      return_state=True,
                                      dropout=0.4,
                                      recurrent_dropout=0.4)

    x, hidden_state = decoder_gru(dec_emb, initial_state=[hidden_state])

    decoder_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(y_voc, activation="softmax"))

    decoder_outputs = decoder_dense(x)

    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam())

    return model


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
