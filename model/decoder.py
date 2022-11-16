import tensorflow as tf
from model.attention import Attention

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.SimpleRNN(self.dec_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # Used for attention
        self.attention = Attention(self.dec_units)

    def call(self, x, hidden, enc_output):

        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)

        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)

        return x, state, attention_weights

def initialize_decoder(vocab_size, embedding_dim, units, batch_size, sample_hidden, sample_output):
    decoder = Decoder(vocab_size, embedding_dim, units * 2, batch_size)

    sample_decoder_output, _, _ = decoder(tf.random.uniform((batch_size, 1)),
                                        sample_hidden, sample_output)

    print ('Decoder output shape: (batch_size, vocab size) ', (sample_decoder_output.shape), '\n')
    return decoder