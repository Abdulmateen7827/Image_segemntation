import sys
import os
from dataclasses import dataclass

import tensorflow as tf
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataPreprocessingConfig:
    image_preprocessed_path: str = os.path.join('artifacts','preprocessed.pkl')

class DataPreprocessing:
    def __init__(self) :
        self.data_preprocessing_config = DataPreprocessingConfig()

    logging.info('function for initializing preprocessing')
    def initializing_preprocessing(self,image_path,mask_path):
        try:
            self.image_path = image_path
            self.mask_path = mask_path

            img = tf.io.read_file(self.image_path)
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)

            mask = tf.io.read_file(self.mask_path)
            mask = tf.image.decode_png(mask, channels=3)
            mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)

            input_image = tf.image.resize(img, (96, 128), method='nearest')
            input_mask = tf.image.resize(mask, (96, 128), method='nearest')

            

            return input_image,input_mask
        
    
        
        except Exception as e:
            raise CustomException(e,sys)
try:        
    logging.info("Saving preprocessor.pkl")
    preprocessor_obj = DataPreprocessing.initializing_preprocessing
    save_object(DataPreprocessingConfig.image_preprocessed_path,preprocessor_obj)
except Exception as e:
    raise CustomException(e,sys)

        