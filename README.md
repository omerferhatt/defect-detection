# Defect Detection and Elliptical Object Localization with DenseNet-169 on subset of DAGM 2007 Defect Dataset

### Task:

Detection of anomalies / disorders of images with convolutional neural networks and detection with an ellipse bounding
boxes. The different and difficult part from normal detection methods is the creation of an elliptical bounding box.
Because it has a parametric and high level function, it is more difficult for the model to predict the elliptical region
than box detection algorithms. In addition to the transfer learning method, both dense and convolutional detection
blocks are placed at the end of the feature extracting blocks. For this reason, in addition to the normal model, the
ellipse shape is estimated by using the regional densities and center point parameters. Instead of the region proposal
network model, which is a more complex structure, direct object localization has been tried and tested whether it can be
detected with a normal feature extractor. Since data augmentation with the elliptic bounding box cannot be performed
with the algorithms used in traditional object detection methods, there is not any data augmentation algorithm is used
on shape. However, some spatial (brightness / contrast / gamma etc.) augmentation algorithms were used to improve
feature extractor performance and reduce model over-fitting.

## Project information:

- Dataset: Subset of DAGM 20007
    - In the dataset, 150 normal and 150 abnormal images were selected for each class. There are 3 classes in total. 20%
      of the images were randomly selected for testing.
    - 900 images in total, 720 for training, 180 for testing
    - All data distributions are equal


- Used model: DenseNet-169
    - Although different state-of-the-art models such as ResNet50 and EfficientNet-B1 have been tried, the best model
      has been DenseNet-169 in terms of parameter number and model success.

