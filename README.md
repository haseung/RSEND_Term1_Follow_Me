### Project: Follow Me!
### Author: Harrison Seung
### Date: 08/18/2018
---
### Writeup / README

### Introduction
The following README is for the Udacity Robotics Software Engineering Nanodegree, Term 1, Project 4, Follow Me!  The goal of the project is to have a Quad Copter drone in simulation track and follow a designated target using a Fully Convolutional Network (FCN).  This report will provide the following:
		○ The network architecture and how it was built with a clear explanation of each network layer including the usage of 1x1 convolutions and fully connected layers
		○ A description of the process for selecting and tuning the neural network parameters
		○ Modifications to the model and data required for use on different objects

### The Network Architecture
The FCN employs a deep learning network that locates a particular human target within an image which allows a simulated quadcopter to follow around the person it detects.  The code consists of 6 sections.

1. Data Collection
2. FCN layers
3. Building the model
4. Training
5. Prediction
6. Evaluation

1. Data Collection

The dataset used for this project was the starting dataset provided by the default github repository from Udacity:

https://github.com/udacity/RoboND-DeepLearning-Project.git

The starting dataset comprises of a training set of 4131 images and a validation set of 1184 images which was sufficient to obtain the required accuracy of 40%.  In order for the model to run the image files must be downloaded and exported into the correct folders under /data/train and /data/validation respectively.

2. FCN layers

Here the layers for the FCN are defined.  The convolutional layers used in the FCN are Separable Convolutional and Bilinear Upsampling.  

 - Separable Convolution Layers: A convolution layer is the process of running a small neural network on an input layer (image), and passing the output layer (image) of different width, height, and depth to the next layer.  A Separable convolutional layer is the process of running a convolution layer over each channel (depth) of an input layer then performing a 1x1 convolution to combine the output channels into an a resultant output layer.  This results in a reduction of parameters used, improving overall runtime performance of the network.  In code, this is provided using the keras function 'SeparableConv2DKeras()'.  Additionally, a batch normalization is performed afterwards to normalize the output of the convolutional layer.  This further optimizes the network by improving the overall network training speed, allowing higher learning rates, and simplifies creation of deeper networks

 - Bilinear Upsampling:  A resampling technique to upsample layers to higher dimensions/resolutions alternative to transposed convolutions.  This is performed after the Separable Convolutional Layers to convert the previous resultant output layers to the original image size.  In code this is provided as 'BilinearUpSampling2D(2,2)(input_layer)', having the effect of increasing the size of the input layer by a factor of 2xRow and 2xCol.   

3. Building the model

[Figure of FCN]

Encoder Block

- The first portion of the FCN is the encoder block.  This consists of 3 operations of the encoder_block().  One instance of the encoder_block() receives an input layer, performs one separable convolution, one batch normalization, and returns the output layer.   The overall effect of the encoder block is converting the original input image from a 160x160x3 layer to a 1x1x128 convolutional layer.  The difference between a Fully CONNECTED Layer and a Fully CONVOLUTIONAL Layer (FCN) is the FCN preserves spatial information as the depth of each layer increases.    

Decoder Block

- The second portion of the FCN is the decoder block.  Here we want to upsample the 1x1 convolutional layer back to the original image size.  This consists of 3 operations of the decoder_block(), the same number of encoder_block().  One instance of the decoder_block() receives two input layers, a small one and a larger one, performs a Bilinear Upsample on the smaller input layer, concatentates the upsampled layer with the larger input layer, performs 2 separable convolutions/batch normalizations, and returns the output layer.  The overall effect of the decoder block is convering the 1x1x128 convolutional layer back to the original image size while retaining additional details from the previous larger layers.

4. Training

Hypyerparameters

learning_rate = 0.01 # trial and error with 0.1 to 0.001 showed 0.01 provided adequate results.

batch_size = 128 # chose a power of 2

num_epochs = 10 # trial and error with 10 to 50 epochs indicate the validation loss plateaus after 10 epochs
[train_vs_val_loss]

steps_per_epoch = 130 # training image set (4131)/ batch_size (128)
validation_steps = 10 # validation set (1184) / batch_size (128)
workers = 2 (default)

5. Prediction

The prediction section provides a visual comparison of the images in a model run.  

[images while following the target]

[images while at patrol without target]

[images while at patrol with target]

6. Evaluation

The accuracy of the network was 40.6%

### Future Enhancments
