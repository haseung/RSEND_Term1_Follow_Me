### Project: Follow Me!
### Author: Harrison Seung
### Date: 08/18/2018
---
### Writeup / README

### Introduction
The following README is for the Udacity Robotics Software Engineering Nanodegree, Term 1, Project 4, Follow Me!  The goal of the project is to have a Quad Copter drone in simulation track and follow a designated target using a Fully Convolutional Network (FCN).  This report will provide the following:

* The network architecture and how it was built with a clear explanation of each network layer including the usage of 1x1 convolutions and fully connected layers
* A description of the process for selecting and tuning the neural network parameters
* Modifications to the model and data required for use on different objects

### The Network Architecture
The FCN employs a deep learning network that locates a particular human target within an image which allows a simulated quadcopter to follow around the person it detects.  The code consists of 6 sections.

* I. Data Collection
* II. FCN layers
* III. Building the model
* IV. Training
* V. Prediction
* VI. Evaluation

### I. Data Collection

The dataset used for this project was the starting dataset provided by the default github repository [here](https://github.com/udacity/RoboND-DeepLearning-Project.git) from Udacity.  The starting dataset comprises of a training set of 4131 images and a validation set of 1184 images which was sufficient to obtain the required accuracy of 40%.  In order for the model to run the image files must be downloaded and exported into the correct folders under /data/train and /data/validation respectively.

### II. FCN layers

Here the layers for the FCN are defined.  The convolutional layers used in the FCN are Separable Convolutional and Bilinear Upsampling.  

* Separable Convolution Layers: A convolution layer is the process of running a small neural network on an input layer (image), and passing the output layer (image) of different width, height, and depth to the next layer.  A Separable convolutional layer is the process of running a convolution layer over each channel (depth) of an input layer then performing a 1x1 convolution to combine the output channels into an a resultant output layer.  This results in a reduction of parameters used, improving overall runtime performance of the network.  In code, this is provided using the keras function 'SeparableConv2DKeras()'.  Additionally, a batch normalization is performed afterwards to normalize the output of the convolutional layer.  This further optimizes the network by improving the overall network training speed, allowing higher learning rates, and simplifies creation of deeper networks

* Bilinear Upsampling:  A resampling technique to upsample layers to higher dimensions/resolutions alternative to transposed convolutions.  This is performed after the Separable Convolutional Layers to convert the previous resultant output layers to the original image size.  In code this is provided as 'BilinearUpSampling2D(2,2)(input_layer)', having the effect of increasing the size of the input layer by a factor of 2xRow and 2xCol.   

### III. Building the model

The FCN consists of two sections, an Encoder Block and a Decoder Block as shown in the figure below.

![FCN Figure](/misc/FCN.JPG)
    

**Encoder Block

* The first portion of the FCN is the encoder block.  This consists of 3 operations of the encoder_block().  One instance of the encoder_block() receives an input layer, performs one separable convolution, one batch normalization, and returns the output layer.   The overall effect of the encoder block is converting the original input image from a 160x160x3 layer to a 1x1x128 convolutional layer.  The difference between a Fully CONNECTED Layer and a Fully CONVOLUTIONAL Layer (FCN) is the FCN preserves spatial information as the depth of each layer increases.    

**Decoder Block

* The second portion of the FCN is the decoder block.  Here we want to upsample the 1x1 CONVOLUTIONAL layer back to the original image size.  This consists of 3 operations of the decoder_block(), the same number of encoder_block().  One instance of the decoder_block() receives two input layers, a small one and a larger one, performs a Bilinear Upsample on the smaller input layer, concatentates the upsampled layer with the larger input layer, performs 2 separable convolutions/batch normalizations, and returns the output layer.  The overall effect of the decoder block is convering the 1x1x128 convolutional layer back to the original image size while retaining additional details from the previous larger layers.

### IV. Training

With the data collected and the FCN code developed, the next step is to train the model.  The hyperparameters available for tuning are listed below with a brief description on how each were selected.

**Hypyerparameters

* learning_rate = 0.01 # The learning rate was tested by examing the results of various learning rates between 0.1 to 0.001.  0.001 and 0.01 both returned similar results and since we are using batch normalization the larger value was selected.

* batch_size = 128 # The batch size was selected as an increasing power of 2 until the accuracy improved to the required accuracy.

* num_epochs = 10 # The num of epcohs was chosen as a comparison of 10 to 50 epochs indicated the validation loss converges after 10 epochs.
    
![train_vs_val_loss](/misc/train_vs_val_loss.jpeg)

* steps_per_epoch = 130 # Equivalent to the training image set (4131)/ batch_size (128)

* validation_steps = 10 # Equivalent to the validation set (1184) / batch_size (128)

* workers = 2 (used default value)

### V. Prediction

The prediction section provides a visual comparison of the images in a model run.  From left to right the images show: original image, sample evaluation result, model result.  Here we can see the model has difficulty identifying the target at far distances and in areas where the target's colors overlap with objects of similar color. 

![images while following the target](/misc/follow_target.jpeg)

Images while following the target

![comparison while at patrol without target](/misc/without_target.jpeg)

Images while at patrol without target

![comparison while at patrol with target](/misc/with_target.jpeg)

Images while at patrol with target

### VI. Evaluation

Initially, the FCN was built using a 2 encode/2 decode layout which resulted in an accuracy of 33%.  Analyzing the false positives and false negatives in the final output and comparing the results with the prediction images, it was clear more spatial information was needed to be passed through the network.  Adding an additional encode/decode layer to the FCN preserved the required spatial information enough for the model to reach an accuracy of 40.6%.  

This model could be used to follow any object however a different dataset would need to be provided as the current dataset is specifically for human target.  If the goal was to follow a dog instead, the follow target and patrol with target datasets would need to include images of dog while the patrol without target would be without.  This will allow the model to learn to classify the characteristics of the dog from the surrounding environment just as the current model does for a human.    

### Future Enhancments

The model could be further improved by collecting additional data of the target at farther distances and zigzagging across regions with similar color tone as these were the areas that generated the most false positives and negatives.  
