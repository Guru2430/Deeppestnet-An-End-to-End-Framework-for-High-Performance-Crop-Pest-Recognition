

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers


    
class Proposed:
    def __init__(self, width_per_group, bottleneck_ratio, num_blocks, units):
        
        # spr
        self.conv1 = tf.keras.layers.Conv1D(width_per_group * bottleneck_ratio, 1, use_bias=False, kernel_initializer='he_normal')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1D(width_per_group * bottleneck_ratio, 3, use_bias=False, kernel_initializer='he_normal')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.shortcut = tf.keras.layers.Conv1D(width_per_group * bottleneck_ratio, 1, use_bias=False, kernel_initializer='he_normal')
        self.bn3 = tf.keras.layers.BatchNormalization()
        
        self.blocks = []
        for i in range(num_blocks):
            self.blocks.append(self.cspbnet_block(width_per_group, bottleneck_ratio))

        self.forward_lstm = layers.LSTM(units, return_sequences=True)
        self.backward_lstm = layers.LSTM(units, return_sequences=True, go_backwards=True)

    
        # Cross stage partial backbone network
        def cspbnet_block(inputs):
            x = self.conv1(inputs)
            x = self.bn1(x)
            x = tf.nn.relu(x)
    
            x = self.conv2(x)
            x = self.bn2(x)
    
            shortcut = self.shortcut(inputs)
            shortcut = self.bn3(shortcut)
            x = tf.concat([x, shortcut], axis=3)
            x = tf.nn.relu(x)
            return x
        
        
        def cspbnet(inputs):
            x = inputs
            for block in self.blocks:
                x = block(x)
            return x
    
        # adaptive attention module
        def attention(queries, values, keys):
            # Queries, values, and keys should be shaped (batch_size, max_length, units)
            # Calculate the attention scores
            scores = tf.matmul(queries, keys, transpose_b=True)
            # Normalize the attention scores
            attention_weights = tf.nn.softmax(scores, axis=-1)
            # Calculate the context vector
            context_vector = tf.matmul(attention_weights, values)
            return context_vector
        
        
        # multi level pyramid structure
        def pyrmid(self, inputs):
            x = tf.keras.activations.tanh()(inputs)
            x = tf.keras.activation.softmax()(x)
            x = tf.keras.activation.softmax()(x)           
            return x
    
        # graph based LSTM
        def gb_bilstm(inputs):
            forward_outputs = self.forward_lstm(inputs)
            backward_outputs = self.backward_lstm(inputs)
            outputs = tf.concat([forward_outputs, backward_outputs], axis=-1)
            return outputs    
        
        self.prediction = tf.contrib.layers.softmax(self.estimation)[:, 1]

        self.cost = tf.add(
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.estimation, labels=self.y)),
            tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
            name="cost")
        
        tf.summary.scalar("cost", self.cost)
        self.merged = tf.summary.merge_all()
               
            
