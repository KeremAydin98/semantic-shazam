import config
from models import *

# Text encoder stack
encoder = Encoder(config.vocab_size, config.embedding_dim, config.enc_units, config.batch_size)

