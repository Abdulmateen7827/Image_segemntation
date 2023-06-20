import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


model_in = tf.keras.models.load_model('artifacts/model.h5')

def welcome():
    return 'Hello'
def preprocess_image(image):
    image = image.resize((128, 96))  # Resize the image to match model input size
    image = np.array(image)  # Convert image to numpy array
    image = image / 255.0  # Normalize the pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def predict_mask(image):

    prediction = model_in.predict(image)
    print(prediction)
    return prediction

def main():
    st.title("Image Segmentation")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Image Segmentation ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload an image",type=['jpg','jpeg','png'])
    image = Image.open(uploaded_image)
    st.image(image,caption='Uploaded_image')


    if st.button('Predict'):
        p = preprocess_image(image)
        p = np.squeeze(p,axis=0)
        input_img = tf.constant(p)
        input_img = tf.data.Dataset.from_tensor_slices([input_img])
        input_img = input_img.batch(32)
        
        pred = model_in.predict(input_img)
        mask = create_mask(pred)
        
        st.image(tf.keras.preprocessing.image.array_to_img(mask), caption='Masked output',use_column_width=True)
        # st.success("The type is {}".format(input_img))

    if st.button('About'):
        st.text('Semantic segmentation on trained data built with streamlit')

if __name__=="__main__":
    main()