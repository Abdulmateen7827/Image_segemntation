import os
import sys

from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate

from src.components.unet import Unet
from src.components.data_preprocessing import DataPreprocessing

from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts','model.pkl')

@dataclass
class ModelDimensionConfig:
    img_height = 96
    img_width = 128
    num_channels = 3

    EPOCHS = 100
    VAL_SUBSPLITS = 5
    BUFFER_SIZE = 500
    BATCH_SIZE = 32

class ModelTrainer:
   
        def __init__(self):
            self.model_trainer_config = ModelTrainerConfig()
            self.dimension = ModelDimensionConfig()

        def initialize_training(self,train_dataset,epoch):
            try:
                unet = Unet.unet_model((self.dimension.img_height,self.dimension.img_width,self.dimension.num_channels))

                unet.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
                
                history = unet.fit(train_dataset,epochs=epoch)
                logging.info("Training completed")

                logging.info('Saved trained model.pkl')
                save_object(self.model_trainer_config.trained_model_file_path,history)

            except Exception as e:
                raise CustomException(e,sys)
