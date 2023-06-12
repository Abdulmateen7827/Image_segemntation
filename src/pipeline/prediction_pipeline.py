import sys
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging
import tensorflow as tf
import matplotlib.pyplot as plt

class PredictPipeline:

    def create_mask(self,pred_mask):
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask[0]
    
    def display(self,display_image):
        plt.figure(figsize=(15, 15))
        plt.title('Predicted mask')
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_image))
        plt.axis('off')
        plt.show()

    def process_img(self,image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        input_image = tf.image.resize(img, (96, 128), method='nearest')

        input_image = tf.constant(input_image)
        input_image = tf.data.Dataset.from_tensor_slices([input_image])
        input_image = input_image.batch(32)

        return input_image


    def predict(self,images):
        logging.info("Predicting")
        # try:
        model_path = 'artifacts/model1.h5'
        model = tf.keras.models.load_model(model_path)
        
        pred = model.predict(self.process_img(images))
        self.display(self.create_mask(pred))


            



# def show_predictions(dataset=None, num=1):
#     """
#     Displays the first image of each of the num batches
#     """
#     if dataset:
#         for image, mask in dataset.take(num):
#             pred_mask = unet.predict(image)
#             display([image[0], mask[0], create_mask(pred_mask)])
#     else:
#         display([sample_image, sample_mask,
#              create_mask(unet.predict(sample_image[tf.newaxis, ...]))])
        
# show_predictions(train_dataset,6)

if __name__=="__main__":
    predict = PredictPipeline()
    predict.predict('/Users/abdulmateen/Downloads/archive/IMAGES/img_0017.jpeg')
