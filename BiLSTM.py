import numpy as np
import tensorflow as tf
from tensorflow import keras
from atten_layer import AttenLayer
from tensorflow.keras import layers,Model

class BiLSTM(Model):
        def __init__(self,
                    word_emb_path,
                    max_seq_len,
                    word_index,
                    embedding_dim,
                    num_classes,
                    hidden_units,
                    kinit,
                    binit,
                    units):
            super(BiLSTM, self).__init__()
            self.MAX_SEQ_LEN = max_seq_len
            self.hidden_units = hidden_units
            self.EMBEDDING_DIM = embedding_dim
            self.NUM_CLASSES = num_classes
            self.units = units

            ## REFERENCE: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
            # read word embeddings from file
            embedding_index = {}
            with open(word_emb_path,encoding = 'utf-8') as f:
                for line in f:
                    values = line..strip("\n").split()
                    word = values[0]
                    embeddings = np.array(values[1:],dtype ="float32")
                    embedding_index[word] = embeddings

            embedding_matrix = np.zeros((len(word_index)+1,self.EMBEDDING_DIM))
            for word,idx in word_index.items():
                embedding_vector = embedding_index.get(word)
                if embedding_vector is not None:
                    # if word is not found embedding is a zero vector
                    embedding_matrix[idx] = embedding_vector

            self.inp_l = layers.Input(shape=(self.MAX_SEQ_LEN))
            self.embeddings = layers.Embedding(input_dim = len(word_index)+1,
                                                output_dim = self.EMBEDDING_DIM,
                                                input_length=self.MAX_SEQ_LEN,
                                                trainable=False,
                                                name="embedding_layer")

            self.concat = layers.Concatenate(name="concateation")
            self.batchnorm1 = layers.BatchNormalization(name="batchnorm1")
            self.batchnorm2 = layers.BatchNormalization(name="batchnorm2")
            self.batchnorm3 = layers.BatchNormalization(name="batchnorm3")
            self.f_lstm = layers.LSTM(self.hidden_units,return_sequences = True,kernel_initializer=kinit,
                                       activity_regularizer="l2",bias_regularizer="l2",bias_initializer=binit,
                                       kernel_regularizer="l2",return_state=True,name="forward_lstm")
            self.b_lstm  = layers.LSTM(self.hidden_units,return_sequences = True,kernel_initializer=kinit,
                                       activity_regularizer="l2",bias_regularizer="l2",bias_initializer=binit,
                                       kernel_regularizer="l2",go_backwards=True,return_state=True,name="backward_lstm")
            self.attenLayer = AttenLayer(self.units,kinit,binit,name="AttentionLayer")
            self.final_layer = layers.Dense(self.NUM_CLASSES,kernel_initializer=kinit,
                                            activity_regularizer="l2",bias_regularizer="l2", kernel_regularizer="l2",
                                            bias_initializer=binit,name="final",activation="softmax")

        def call(self,inputs):
            """
                Arguments: input - tokenized input padded to maximum sequence length
            """

            x = self.embeddings(inputs) #shape of x after embedding is (batch_size, max_seq_len, embedding_dim)
            x = self.batchnorm1(x) #shape after batch normalization is (batch_size, max_seq_len, embedding_dim)
            fhidden_states, flast_state, flast_current_state =  self.f_lstm(x)
            bhidden_states, blast_state, blast_current_state = self.b_lstm(x)

            #fhidden_states, bhidden_states shape (batch_size, max_seq_len,hidden_units)
            #flast_hidden_state,  blast_hidden_state shape (batch_size,hidden_units)

            hidden_states = self.concat(([fhidden_states,bhidden_states]))
            hidden_states = self.batchnorm2(hidden_states)
            # hidden_states shape (batch_size,max_seq_len,2*hidden_units)

            last_state = self.concat(([flast_state,blast_state])) #shape of last_state is (batch_size,2*hidden_units)
            context_vector,attention_weights = self.attenLayer(hidden_states)

            #context_represent = tf.math.add(context_vector, last_state) #shape of context_prepresent is (batch_size,2*hidden_units)
            context_represent = self.batchnorm3(context_vector)
            logits_ = self.final_layer(context_represent) # shape of logits_ (batch_size,num_classes)
            return logits_
