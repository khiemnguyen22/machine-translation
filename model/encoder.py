import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.rnn = tf.keras.layers.GRU(self.enc_units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
        self.rnn = tf.keras.layers.Bidirectional(self.rnn)

    # Encoder network comprises an Embedding layer followed by a GRU layer
    def call(self, x, hidden):
        x = self.embedding(x)
        output, forward_state, backward_state = self.rnn(x, initial_state=hidden)
        return output, tf.keras.layers.Concatenate()([forward_state, backward_state])

    # To initialize the hidden state
    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_sz, self.enc_units)) for i in range(2)]
  
def initialize_encoder(vocab_size, embedding_dim, units, batch_size, example_input_batch):
    encoder = Encoder(vocab_size, embedding_dim, units, batch_size)

    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

    print ('Encoder output shape: (batch size, sequence length, units) ',sample_output.shape)
    print ('Encoder Hidden state shape: (batch size, units) ',sample_hidden.shape)

    return encoder, sample_hidden, sample_output