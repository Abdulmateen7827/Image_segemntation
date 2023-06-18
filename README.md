# Image Semantic segmentation

This repository entails code and resources from a machine learning competition on semantic Image segmentation.
This project is hosted on kaggle and the dataset contains images and their respective masks. Semantic image segmentation provides pixel wise compared to that of object localization in which a bounding box is applied on the detected image. It has various use cases like in self driving cars and biomedics

The goal of this project is to build a model to segment images on the data trained on. 

# Model architecture
The semantic segmentation model is based on the U-Net architecture, widely used for image segmentation tasks. The U-Net model consists of an encoder path that captures the context of the image and a decoder path that performs upsampling and produces the segmentation mask.

![Alt text](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

Due to computational reason I reduced the model complexity by reducing the filters to 32 (original is 64) therefore resulting in a less accurate prediction. This project just serves as a sneak peek to what the original model can achieve.


If you are interested in the competition or just want to learn more about the project, you can check the resources in this repositiory.


# Live Demo
