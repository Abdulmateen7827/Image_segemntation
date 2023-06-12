import os
import sys

from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate



class Blocks:
    try:
        def conv_block(self,inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):

            """
            Convolutional downsampling block
            
            Arguments:
                inputs -- Input tensor
                n_filters -- Number of filters for the convolutional layers
                dropout_prob -- Dropout probability
                max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
            Returns: 
                next_layer, skip_connection --  Next layer and skip connection outputs
            """
            self.inputs = inputs
            self.n_filters = n_filters
            self.dropout_prob = dropout_prob
            self.max_pooling = max_pooling

            conv = Conv2D(filters=self.n_filters, 
                        kernel_size=3,     
                        activation='relu',
                        padding='same',
                        kernel_initializer='he_normal')(self.inputs)
            conv = Conv2D(filters=self.n_filters, 
                        kernel_size=3,    
                        activation='relu',
                        padding='same',
                        kernel_initializer='he_normal')(conv)
            
            if self.dropout_prob > 0:
                conv = Dropout(self.dropout_prob)(conv)
                
                
            if self.max_pooling:
                next_layer = MaxPooling2D(2,2)(conv)
                
            else:
                next_layer = conv
                
            skip_connection = conv
            
            return next_layer, skip_connection
        
        def upsampling_block(self,expansive_input, contractive_input, n_filters=32):
            """
            Convolutional upsampling block
            
            Arguments:
                expansive_input -- Input tensor from previous layer
                contractive_input -- Input tensor from previous skip layer
                n_filters -- Number of filters for the convolutional layers
            Returns: 
                conv -- Tensor output
            """
            self.expansive_inp = expansive_input
            self.conc_inp = contractive_input
            self.n_filt = n_filters

            up = Conv2DTranspose(
                        self.n_filt,   
                        3,    
                        strides=(2,2),
                        padding='same')(self.expansive_inp)
            
            # Merge the previous output and the contractive_input
            merge = concatenate([up, self.conc_inp], axis=3)
            conv = Conv2D(self.n_filt,   
                        3,    
                        activation='relu',
                        padding='same',
                        kernel_initializer='he_normal')(merge)
            conv = Conv2D(self.n_filt,  
                        3,   
                        activation='relu',
                        padding='same',
                        kernel_initializer='he_normal')(conv)
            
            return conv
        
    except Exception as e:
        raise CustomException(e,sys)

block = Blocks()

class Unet:
    try:
        def unet_model(self,input_size=(96, 128, 3), n_filters=32, n_classes=59):
            """
            Unet model
            
            Arguments:
                input_size -- Input shape 
                n_filters -- Number of filters for the convolutional layers
                n_classes -- Number of output classes
            Returns: 
                model -- tf.keras.Model
            """
            inputs = Input(input_size)
            
            # Contracting Path (encoding)
        
            cblock1 = block.conv_block(inputs=inputs, n_filters=n_filters)

            cblock2 = block.conv_block(cblock1[0], n_filters=n_filters*2)
            cblock3 = block.conv_block(cblock2[0], n_filters=n_filters*4)
            cblock4 = block.conv_block(cblock3[0], n_filters=n_filters*8, dropout_prob=0.3) 

            cblock5 = block.conv_block(cblock4[0], n_filters=n_filters*16, dropout_prob=0.3, max_pooling=False) 
            
            
            # Expanding Path (decoding)
            
            ublock6 = block.upsampling_block(cblock5[0], cblock4[1],  n_filters= n_filters*8)
            # Chaining the output of the previous block as expansive_input and the corresponding contractive block output.

            ublock7 = block.upsampling_block(ublock6, cblock3[1],  n_filters= n_filters*4)
            ublock8 = block.upsampling_block(ublock7, cblock2[1],  n_filters= n_filters*2)
            ublock9 =block.upsampling_block(ublock8, cblock1[1],  n_filters= n_filters)
            

            conv9 = Conv2D( n_filters,
                        3,
                        activation='relu',
                        padding='same',
                        kernel_initializer='he_normal')(ublock9)

            
            conv10 = Conv2D(n_classes, 1, padding='same')(conv9)
            
            
            model = tf.keras.Model(inputs=inputs, outputs=conv10)

            return model
        
    except Exception as e:
        raise CustomException(e,sys)
