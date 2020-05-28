#!-*- encoding=utf-8 -*-
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence
from keras.preprocessing.text import Tokenizer
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback
from keras.initializers import Ones, Zeros

import numpy as np
import random
import sys
import os
import json

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import keras.backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True 
sess = tf.Session(config=config)

class Position_Embedding(Layer):
    
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size 
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)
        
    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size,seq_len = K.shape(x)[0],K.shape(x)[1]
        position_j = 1. / K.pow(10000., \
                                 2 * K.arange(self.size / 2, dtype='float32' \
                               ) / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:,:,0]), 1)-1 #K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)
        
    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2]+self.size)

class TiedEmbeddingsTransposed(Layer):
   """Layer for tying embeddings in an output layer.
   A regular embedding layer has the shape: V x H (V: size of the vocabulary. H: size of the projected space).
   In this layer, we'll go: H x V.
   With the same weights than the regular embedding.
   In addition, it may have an activation.
   # References
       - [ Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
   """

   def __init__(self, tied_to=None,
                activation=None,
                **kwargs):
       super(TiedEmbeddingsTransposed, self).__init__(**kwargs)
       self.tied_to = tied_to
       self.activation = activations.get(activation)

   def build(self, input_shape):
       self.transposed_weights = K.transpose(self.tied_to.weights[0])
       self.built = True


   def compute_output_shape(self, input_shape):
       return input_shape[0], input_shape[1], K.int_shape(self.tied_to.weights[0])[0]

   def call(self, inputs, mask=None):
       output = K.dot(inputs, self.transposed_weights)
       if self.activation is not None:
           output = self.activation(output)
       return output


   def get_config(self):
       config = {'activation': activations.serialize(self.activation)
                 }
       base_config = super(TiedEmbeddingsTransposed, self).get_config()
       return dict(list(base_config.items()) + list(config.items()))
  
class Attention(Layer):
    #uni-directional self attention for long-range text generation
    def __init__(self, nb_head, size_per_head, max_len, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        self.max_len = max_len
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ', 
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK', 
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV', 
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
        
    def Mask(self, inputs):
        mask = K.eye(self.max_len) #[ml, ml]
        mask = K.cumsum(mask, 1) #[ml,ml]
        mask =  K.expand_dims(mask, axis=0) #[bs, ml, ml]
        
        
        eye = K.eye(self.max_len)
        eye = K.expand_dims(eye, axis=0)
        mask = mask - eye
        
        mask = K.expand_dims(mask, axis=1) #[1,1, ml,ml]
        mask = K.permute_dimensions(mask, (0,3,2,1))
     
        return inputs - mask * 1e12
                
    def call(self, x):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x

        Q_seq = K.dot(Q_seq, self.WQ) #[bs, ml, output_dim]
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head)) #[bs, ml, nb_head, size_per_head]
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))  #[bs, nb_head, ml, size_per_head]
        K_seq = K.dot(K_seq, self.WK) #[bs, ml, output_dim]
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3)) #[bs, nb_head, ml, size_per_head]
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3)) #[bs, nb_head, ml, size_per_head]
 
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5 #[bs, nb_head, ml, ml]
        A = K.permute_dimensions(A, (0,3,2,1)) #[bs, ml, ml, nb_head]
        A = self.Mask(A) 
        A = K.permute_dimensions(A, (0,3,2,1)) #[bs, nb_head, ml, ml]
        A = K.softmax(A)

        O_seq = K.batch_dot(A, V_seq, axes=[3,2]) #[bs, nb_head, ml, size_per_head]
    #    print(O_seq.shape)
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3)) #[bs, ml, nb_head, size_per_head]
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
    #    O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)
    
class LayerNormalization(Layer):
   """
       Implementation according to:
           "Layer Normalization" by JL Ba, JR Kiros, GE Hinton (2016)
   """

   def __init__(self, epsilon=1e-8, **kwargs):
       self._epsilon = epsilon
       super(LayerNormalization, self).__init__(**kwargs)
   
   def compute_output_shape(self, input_shape):
       return input_shape
   
   def build(self, input_shape):
       self._g = self.add_weight(
           name='gain', 
           shape=(input_shape[-1],),
           initializer=Ones(),
           trainable=True
       )
       self._b = self.add_weight(
           name='bias', 
           shape=(input_shape[-1],),
           initializer=Zeros(),
           trainable=True
       )
       super(LayerNormalization, self).build(input_shape)
       
   def call(self, x):
       mean = K.mean(x, axis=-1)
       std = K.std(x, axis=-1)

       if len(x.shape) == 3:
           mean = K.permute_dimensions(
               K.repeat(mean, x.shape.as_list()[-1]),
               [0,2,1]
           )
           std = K.permute_dimensions(
               K.repeat(std, x.shape.as_list()[-1]),
               [0,2,1] 
           )
           
       elif len(x.shape) == 2:
           mean = K.reshape(
               K.repeat_elements(mean, x.shape.as_list()[-1], 0),
               (-1, x.shape.as_list()[-1])
           )
           std = K.reshape(
               K.repeat_elements(mean, x.shape.as_list()[-1], 0),
               (-1, x.shape.as_list()[-1])
           )
       
       return self._g * (x - mean) / (std + self._epsilon) + self._b
    
