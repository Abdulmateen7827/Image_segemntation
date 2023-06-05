import os
import sys
from src.logger import logging
from src.exception import CustomException
import shutil

import pandas as pd
import numpy as np
from dataclasses import dataclass

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
            image_list = os.listdir(self.data_ingestion_config.images_data)
            mask_list = os.listdir(self.data_ingestion_config.masks_data)

            image_list = [self.data_ingestion_config.images_data+"/"+i for i in image_list]
            mask_list = [self.data_ingestion_config.masks_data+"/"+i for i in mask_list]

            
            return (
                image_list,
                mask_list
            )
            

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()


