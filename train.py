import argparse
from tabnanny import check
from data.gen_dataset import *
from data.preprocessing import preprocess_line
from model.encoder import *
from model.decoder import *
from model.attention import *
from translate import *
import time
import os

def loss_function(real, pred, loss_object):
      mask = tf.math.logical_not(tf.math.equal(real, 0))
      loss_ = loss_object(real, pred)

      mask = tf.cast(mask, dtype=loss_.dtype)
      loss_ *= mask

      return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden, encoder, decoder, optimizer, out_tokenizer, batch_size, loss_object):
      loss = 0

      with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([out_tokenizer.word_index['<start>']] * batch_size, 1)

        for t in range(1, targ.shape[1]):


          predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)


          loss += loss_function(targ[:, t], predictions, loss_object)


          dec_input = tf.expand_dims(targ[:, t], 1)

      batch_loss = (loss / int(targ.shape[1]))

      variables = encoder.trainable_variables + decoder.trainable_variables

      gradients = tape.gradient(loss, variables)

      optimizer.apply_gradients(zip(gradients, variables))
      return batch_loss

def main(args):
      if args.dataset == 'europarl':
            trainX, trainY, testX, testY, in_tokenizer, out_tokenizer, in_vocab_size, out_vocab_size= gen_dataset(args.input_data, args.output_data, args.num_samples)
      if args.dataset == 'manythings':
            trainX, trainY, testX, testY, in_tokenizer, out_tokenizer, in_vocab_size, out_vocab_size= gen_manythings_dataset(args.path_manythings, args.num_samples, 'english', 'french')
     
      # model parameters
      batch_size = args.batch_size
      buffer_size = len(trainX)
      steps_per_epoch = len(trainY) // batch_size
      embedding_dim = args.embedding_dim
      units = args.units
      epochs = args.epochs

      dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY)).shuffle(buffer_size)
      dataset = dataset.batch(batch_size, drop_remainder=True)
      example_input_batch, example_target_batch = next(iter(dataset))

      encoder, sample_hidden, sample_output = initialize_encoder(in_vocab_size, embedding_dim, units, batch_size, example_input_batch)
      attention = initialize_attention(args.attention_units, sample_hidden, sample_output)
      decoder = initialize_decoder(out_vocab_size, embedding_dim, units, batch_size, sample_hidden, sample_output)

      optimizer = tf.keras.optimizers.Adam()

      loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True, reduction='none')

      checkpoint_dir = './training_checkpoints'
      checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
      checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                      encoder=encoder,
                                      decoder=decoder)

      if args.train:
            for epoch in range(epochs):
                  start = time.time()

            # Initialize the hidden state
                  enc_hidden = encoder.initialize_hidden_state()
                  total_loss = 0

                  # Loop through the dataset
                  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):

                        # Call the train method
                        batch_loss = train_step(inp, targ, enc_hidden, encoder, decoder, optimizer, out_tokenizer, batch_size, loss_object)

                        # Compute the loss (per batch)
                        total_loss += batch_loss

                        if batch % 100 == 0:
                              print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                        batch,
                                                                        batch_loss.numpy()))
                  # Save (checkpoint) the model every 2 epochs
                  if (epoch + 1) % 2 == 0:
                        checkpoint.save(file_prefix = checkpoint_prefix)

                  # Output the loss observed until that epoch
                  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                      total_loss / steps_per_epoch))
                  
                  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))    
      else:
            input_sentence = input('Input sentences: ')
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
            translate(input_sentence, encoder, decoder, units, trainY.shape[1], trainX.shape[1], in_tokenizer, out_tokenizer)

      

if __name__ == '__main__':
      parser = argparse.ArgumentParser(description='model training')

      parser.add_argument('--dataset', type=str, default='manythings')

      parser.add_argument('--path_manythings', type=str, default='fra.txt')

      parser.add_argument('--input_data', type=str, default='english.pkl')

      parser.add_argument('--output_data', type=str, default='french.pkl')

      parser.add_argument('--num_samples', type=int, default=50000)

      parser.add_argument('--batch_size', type=int, default=64)

      parser.add_argument('--embedding_dim', type=int, default = 256)

      parser.add_argument('--units', type=int, default = 512)

      parser.add_argument('--attention_units', type=int, default = 10)

      parser.add_argument('--epochs', type=int, default = 15)

      parser.add_argument('--train', type=bool, default = True)

      main(parser.parse_args())
