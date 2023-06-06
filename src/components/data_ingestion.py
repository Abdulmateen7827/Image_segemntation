import os
import sys
from src.logger import logging
from src.exception import CustomException
import shutil
import tensorflow as tf

import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.components.data_preprocessing import DataPreprocessing


path = '/Users/abdulmateen/Downloads/car-segmentation' 

@dataclass
class DataIngestionConfig:

    images_data: str = os.path.join(path, 'images')
    masks_data: str = os.path.join(path, 'masks')
logging.info('Returned images and masks in lists')
class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            image_list_ = os.listdir(self.data_ingestion_config.images_data)
            mask_list_ = os.listdir(self.data_ingestion_config.masks_data)

            image_list = [self.data_ingestion_config.images_data+"/"+i for i in image_list_]
            mask_list = [self.data_ingestion_config.masks_data+"/"+i for i in mask_list_]
            

            image_filenames = tf.constant(image_list)
            mask_filenames = tf.constant(mask_list)

            dataset = tf.data.Dataset.from_tensor_slices((image_filenames, mask_filenames))
            logging.info("Data ingestion is completed")

            return dataset
            # for image, mask in dataset.take(1):
            #     print(image)
            #     print(mask)


            
            

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    dataset = obj.initiate_data_ingestion()

    preprocessor = DataPreprocessing()
    dataset = dataset.map(preprocessor.initializing_preprocessing)
    logging.info("preprocessing completed")



    


