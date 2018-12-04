#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 10:39:37 2018

@author: stenatu
"""
from __future__ import division, absolute_import, print_function
import tensorflow as tf
tf.enable_eager_execution()

print("Make sure Tensorflow Version > 1.10. Mine is {}".format(tf.__version__))

import unicodedata
import re
import numpy as np
import os
import time

# If file is not downloaded, download the file using Keras as below
#path_to_zip = tf.keras.utils.get_file('ron-eng.zip', origin = 'http://www.manythings.org/anki/ron-eng.zip',
#                                     extract = True)
path_to_zip = os.getcwd()
path_to_file = path_to_zip + '/ron-eng/ron.txt'

# Clean the dataset by removing special characters

def unicode_to_ascii(a):
    return ''.join(c for c in unicodedata.normalize('NFD', a)
                   if unicodedata.category(c) != 'Mn')

def preprocess_sentence(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!,?])", r" \l ", s)
    s = re.sub(r'[" "]+', " ", s)
    
    s = re.sub(r"[^a-zA-Z?.!,]+", " ", s)
    s = s.rstrip().strip()
    
    s = '<start> ' + s + ' <end>' # add start and end tokens - note the spacing between start and end tokens 
    return s

# preprocess the entire text and get the dataset for training. Use num examples to reduce the number of
# training steps. Naturally more the training examples, better the model.    

def create_dataset(path, num_examples = 10000):
    text = open(file = path_to_file, encoding = 'UTF-8').read().strip().split('\n')
    if num_examples < 10000:
        word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in text]
    else:
        word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in text[:10000]]

    return word_pairs

pairs = create_dataset(path_to_zip)

# create a Language class that you can use in different contexts
class LanguageIndex():
    '''Creates a mapping between words and numbers. word --> index (e.g. "cat" --> 3) and vice-versa.
    Use this class more generally for text generation models.'''
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        
        self.create_index()
    
    def create_index(self):
        ''' create a vocabulary and an word to index dictionary and vice-versa'''
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))
        
        self.vocab = sorted(self.vocab)
        
        self.word2idx['<pad>'] = 0
        self.idx2word[0] = '<pad>' 
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1 #because 0 is reserved for pad.
            self.idx2word[index] = word
    # this returns a LanguageIndex object. Aternatively put return statements 
    # for the function and just call the create_index functionn. 
        

# Load the dataset for english to Romanian translation
def max_length(tensor):
    return max([len(t) for t in tensor])
    
def load_dataset(pairs):
    inp_lang = LanguageIndex(en for en, rom in pairs)
    tar_lang = LanguageIndex(rom for en, rom in pairs)
      
      # get the index tensor.
    input_tensor = [[inp_lang.word2idx[word] for word in en.split(' ')] for en, rom in pairs]
    target_tensor = [[tar_lang.word2idx[word] for word in rom.split(' ')] for en, rom in pairs]
    
      # each of these sublists are of a different length. So we pad them by max_length
    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)
    
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, maxlen = max_length_inp
                                                                  , padding = 'post')
      
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, maxlen = max_length_tar
                                                                  , padding = 'post')
    
    
    return inp_lang, tar_lang, input_tensor, target_tensor, max_length_inp, max_length_tar
    
  
inp_lang, tar_lang, input_tensor, target_tensor, max_length_inp, max_length_tar = load_dataset(pairs)
print(list(inp_lang.word2idx)[:10])
#print(list(tar_lang.word2idx)[:5])
    
# Check that the tensors have the corect shape
print("Max Length English = {}, Max Length Romanian {}".format(max_length_inp, max_length_tar))
print("Shape of input tensor = {}".format(np.shape(input_tensor)))   
print("Shape of target tensor = {}".format(np.shape(target_tensor))) 

# Create the test, train split to train the model. USe a random 80-20 split here.

from sklearn.model_selection import train_test_split

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                               target_tensor,
                                                                                               test_size = 0.2
                                                                                                )

## Create a tf.data dataset which takes data in batches for training. 

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
N_BATCH = BUFFER_SIZE//BATCH_SIZE

embedding_dim = 256
units = 10
vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(tar_lang.word2idx)

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

# Check that the dataset shape object has the right dimensions -- inputs of Batch_size, max_english and output 
# Batch_size, max_romanian

print("The dataset shape is ----> {}".format(dataset))

# Model 

def gru(units):
    ''' If you have a GPU , this model defaults to a CuDNNGRU or else a GRU '''
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(units, 
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer ='glorot_uniform')
        
        
    else:
        return tf.keras.layers.GRU(units, 
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_activation='sigmoid',
                                        recurrent_initializer='glorot_uniform')

class Encoder(tf.keras.Model):
    ''' Writes the Encoder Class used for training '''
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder,self).__init__() #super class as they both subclass tf.keras.Model
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.enc_units = enc_units
        self.batch_size = batch_size
        self.gru = gru(self.enc_units)
    
    def call(self, x, hidden):
        # input vector has shape (batch_size, max_input_length)
        x = self.embedding(x)
        # embedded vector has shape (batch_size, max_input_length, embedding_dim)
        output, state = self.gru(x, initial_state = hidden)
        # output_shape = [batch_size, max_input_length, enc_units]
        # state = [batch_size, enc_units]
        return output, state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))
    
    
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.dec_units = dec_units
        self.batch_size = batch_size
        self.gru = gru(units = self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)
        
        # define the attention weights for Bahndau attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V =tf.keras.layers.Dense(1)
    

    def call(self, x, hidden, enc_output):
        
        #enc_output shape = [batch_size, max_input_length, enc_units].
        
        # In below we want to compute the attention weights for each word which comes in the 
        # max_input_length dimension. FInally we want to sum over that dimension to get the context
        # vector.
        
        # first implement attention mechanism
        
        # compute score from hidden state of decoder and the current output of the encoder
        # at that timestep.
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        # this is essential so that hidden has the same shape as output of the encoder.
        
        #score_shape = (batch_size, max_length, 1)
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
        # the output of the score is [batch_size, max_length, 1]
        
        # from the score construct the attention weights
        attention_weights = tf.nn.softmax(score, axis = 1)
        
        #sum over all the attention weights over all the hidden states 
        # after unrolling through time.
        #context_vector has shape (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis = 1)
        
        
        # code to get the hidden states of the target vectors
        x = self.embedding(x)
        
        # x has shape -- [batch_size, 1, embedding_dim] -- 1 because each word is being compared
        # to the input to compute the attention vector
        
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis = -1)
        # x after concat has shape [batch_size, 1, hidden_size + embedding_dim]
        output, state = self.gru(x)


        output = tf.reshape(output, (-1, output.shape[2]))
        # output shape  = [batch_size*1, batch_size+embedding_dim]
        
        x = self.fc(output)
        # shape of x = [batch_size, vocab_size]
        
        return x, state, attention_weights
    
    def initialize_hidden_state(self):
        tf.zeros([self.batch_size, self.dec_units])
    

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
        
# define the Optimizer and the Loss
optimizer = tf.train.AdamOptimizer()

def loss_function(real, pred):
    mask = 1-np.equal(real, 0)
    loss_red = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = real, logits=pred)*mask
    return tf.reduce_mean(loss_red)

# Checkpoints for Model Serving

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


EPOCHS = 5

for epoch in range(EPOCHS):
    start = time.time()
    hidden = encoder.initialize_hidden_state()
    total_loss = 0
    
    for (batch, (inp, targ)) in enumerate(dataset):
       loss= 0
       
       with tf.GradientTape() as tape:
           enc_output, enc_hidden= encoder(inp, hidden)
           
           dec_hidden = enc_hidden
           
           # initial input is pad - multiply by Batchsize to create that vector, expand dims to match the needed input dimensions
           dec_input = tf.expand_dims([tar_lang.word2idx['<pad>']]*BATCH_SIZE, 1)
           #print(targ.shape)
           for t in range(1, targ.shape[1]):
               #generate predictions from the decoder model -- which uses decoder_input, hidden and enc_output
               predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
               
               loss += loss_function(targ[:, t], predictions)
               
               # teacher forcing -- put the target word as input to decoder
               
               dec_input = tf.expand_dims(targ[:, t], 1)
              
       batch_loss = (loss/int(targ.shape[1]))
       
       total_loss += batch_loss
       
       variables = encoder.variables + decoder.variables
       
       gradients = tape.gradient(loss, variables)
       
       optimizer.apply_gradients(zip(gradients, variables))
       
       if batch % 100 == 0:
           print('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1, batch, batch_loss.numpy()))
           
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
    
    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / N_BATCH))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))         
           
    
def evaluate(sentence,encoder, decoder, inp_lang, tar_lang, max_length_inp, max_length_tar):
    sentence = preprocess_sentence(sentence)
    
    inputs = [inp_lang.word2idx[i] for i in sentence.split(' ')]
    print(inputs)
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen = max_length_inp
                                                           , padding = 'post')
    inputs = tf.convert_to_tensor(inputs)
    
    result = ''
    
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    
    dec_hidden = enc_hidden
    # Make a batch of 1
    dec_input = tf.expand_dims([tar_lang.word2idx['<start>']], 0)
    
    for t in range(max_length_tar):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_out)
        
        predicted_id = tf.argmax(predictions[0]).numpy()

        result += tar_lang.idx2word[predicted_id] + ' '

        if tar_lang.idx2word[predicted_id] == '<end>':
            return result, sentence
        
    dec_input = tf.expand_dims([predicted_id], 0) # put the predicted id back into the model
    
    return result,sentence

def translate(sentence, encoder, decoder, inp_lang, tar_lang, max_length_inp, max_length_tar):
    result, sentence = evaluate(sentence, encoder, decoder, inp_lang, tar_lang, max_length_inp, max_length_tar)
        
    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(result))
    
# Restore the model from the latest checkpoint and evaluate on some sentences
    
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

translate('This is an aberration', encoder, decoder, 
        inp_lang, tar_lang, max_length_inp, max_length_tar)
             
       
       
          
        
        
        
        
