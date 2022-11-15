import config
from models import *

# Text encoder stack
encoder = Encoder(config.vocab_size, config.embedding_dim, config.enc_units, config.batch_size)

train_dataset, val_dataset, inp_lang, targ_lang

# Test decoder stack
decoder = Decoder(vocab_tar_size, config.embedding_dim, units, BATCH_SIZE)
