import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AttenLayer(layers.Layer):

    """
       Uses hidden states of the BiLSTM/BiGRU to compute
       attention score and word importance
    """

    def __init__(self,units,kinit,binit,**kwargs):
        super(AttenLayer, self).__init__(**kwargs)
        self.units = units
        self.W_h = layers.Dense(self.units,
                                kernel_initializer=kinit, bias_initializer=binit,
                                kernel_regularizer="l2",bias_regularizer="l2",activity_regularizer='l2')
        self.V = layers.Dense(1,kernel_initializer=kinit,use_bias=False,
                              kernel_regularizer="l2",activity_regularizer='l2')

    def  call(self,hidden_states):
        """
            Compute weighted representation of the sentence using attention mechanism

            Params:
                   hidden_states: hidden states of the BiLSTM/BiGRU of all time steps
                                  shape (batch_size,max_seq_len,2*hidden_units)

            Return:
                   contex_vector: contextual representaion of the input sentence computed
                                  by considering the weighted sum of hidden_state of each time step
                                  shape (batch_size, 2*hidden_units)

                   attention_weights: weighted scores for words in the input sentence, representing
                                      the relative importance of each word in sentiment calssification
                                      shape (batch_size, max_seq_len, 1)
        """
        #we get a tensor of shape (batch_size, max_seq_len, units) after applying self.W_h to hidden_states
        #the shape of tensor before applying self.V is (batch_size, max_seq_len, units)
        #we get a tensor of shape (batch_size,max_seq_len,1) after applying self.V i.e shape of score is (batch_size,max_seq_len,1)

        score = self.V(tf.nn.tanh(self.W_h(hidden_states)))
        attention_weights = tf.nn.softmax(score,axis = 1) # shape of attention is (batch_size,max_seq_len,1)

        context_vector = attention_weights * hidden_states #shape of context_vector is (batch_size, max_seq_len, 2*hidden_states)
        #sum across time_steps/max_seq_len, shape of context_vector after sum is (batch_size,2*hidden_states)
        context_vector = tf.reduce_sum(context_vector,axis = 1)

        return context_vector, attention_weights

    def get_config(self):
        config = super(AttenLayer, self).get_config()
        config.update({"units":self.units,"W_h":self.W_h, "V":self.V})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
