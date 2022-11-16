import tensorflow as tf 

class Attention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(Attention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):

    query_with_time_axis = tf.expand_dims(query, 1)

    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    attention_weights = tf.nn.softmax(score, axis=1)

    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

def initialize_attention(units, sample_hidden, sample_output):
    attention_layer = Attention(units)
    attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
    
    print('Attention result shape: (batch size, units) ', (attention_result.shape))
    print('Attention weights shape: (batch_size, sequence_length, 1) ',(attention_weights.shape))
    
    return attention_layer, sample_hidden, sample_output
