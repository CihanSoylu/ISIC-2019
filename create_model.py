import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score

class clf(tf.keras.Model):
    
    def __init__(self, regularization, unit_sizes):
        super(clf, self).__init__()
        
        self.regularization = regularization
        
        self.depth = len(unit_sizes)
        
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.dropout_1 = tf.keras.layers.Dropout(0.5)
        self.dense_1 =  tf.keras.layers.Dense(unit_sizes[0], activation = 'relu',
                             kernel_regularizer = regularization)
        
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.dropout_2 = tf.keras.layers.Dropout(0.5)
        self.dense_2 = tf.keras.layers.Dense(unit_sizes[1], activation = 'relu',
                                 kernel_regularizer = regularization)
        
        if self.depth == 3:
            self.batch_norm_3 = tf.keras.layers.BatchNormalization()
            self.dropout_3 = tf.keras.layers.Dropout(0.5)
            self.dense_3 = tf.keras.layers.Dense(unit_sizes[2], activation = 'relu',
                                             kernel_regularizer = regularization)
        
        self.output_layer = tf.keras.layers.Dense(8, activation='softmax', 
                                        name = 'output_layer')
        
    
    def call(self, inputs):
        
        x = self.batch_norm_1(inputs)
        x = self.dropout_1(x)
        x = self.dense_1(x)
        
        x = self.batch_norm_2(x)
        x = self.dropout_2(x)
        x = self.dense_2(x)
        
        if self.depth == 3:
            x = self.batch_norm_3(x)
            x = self.dropout_3(x)
            x = self.dense_3(x)
        
        return self.output_layer(x)



def get_model(regularization, unit_sizes):
    
    model = clf(regularization, unit_sizes)
    
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                  optimizer=tf.keras.optimizers.RMSprop(), 
                  metrics=['accuracy'])

    return model

