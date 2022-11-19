import config
from models import *
import tensorflow as tf

# Text encoder stack
encoder = Encoder(config.vocab_size, config.embedding_dim, config.enc_units, config.batch_size)

lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')
lang_tokenizer.fit_on_texts(lang)

train_dataset, val_dataset, inp_lang, targ_lang = dataset_creator.call(config.num_examples, config.BUFFER_SIZE,
                                                                       config.BATCH_SIZE)



