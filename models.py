## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # not adding padding - therefore need to ensure images are divisible by 5 to avoid columns being lost (as pyTorch leaves 
        # it up to user to pad)
        # see: http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(1, 32, 5)
        self.conv3 = nn.Conv2d(1, 32, 5)
        # kernel size of (2,2) and since left out stride it defaults to the size of the kernel. Note: putting a pooling layer after
        # every convolutional layer just like we did in Keras CV capstone
        self.maxPool1 = nn.MaxPool2D(2)
        self.maxPool2 = nn.MaxPool2D(2)
        self.maxPool3 = nn.MaxPool2D(2)

        self.drop1 = nn.Dropout2d(p=.3)
        self.drop2 = nn.Dropout2d(p=.15)
        self.drop3 = nn.Dropout2d(p=.10)
        self.drop4 = nn.Dropout2d(p=.10)

        # note: this is how we make the 'Dense' (fully connected) layers like we did in Keras. There are called Linear here because remember
        # dense layer is a layer that is a linear operation of all the input vectors. We then will add on in the 'forward' function the 
        # activation function we want to use on this layer. Remember the last layer with the output of 126 values will have no activation 
        # function because we aren't classifying but really more doing a regression. While the other ones we are classifying features within 
        # the face to help us determine facial keypoints
        self.linear1 = nn.Linear(5000)
        self.linear2 = nn.Linear(2500)
        self.linear3 = nn.Linear(500)
        self.linear4 = nn.Linear(136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and layers to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # note: X starts of as the input image. And then just like we do in keras we keep using it as the output of each 
        # subsequent layer
        x = self.maxPool1(F.relu(self.conv1(x)))
        x = self.maxPool2(F.relu(self.conv2(x)))
        x = self.maxPool3(F.relu(self.conv3(x)))
        x = self.drop1(x)
        # this should be like a Flatten layer. Where putting just the argument '-1' just takes the input layer dimensions and then 
        # just flattens it accordingly
        x = x.view(-1)
        x = F.relu(self.linear1(x))
        x = self.drop2(x)
        x = F.relu(self.linear2(x))
        x = self.drop3(x)
        x = F.relu(self.linear3(x))
        x = self.drop4(x)
        x = F.relu(self.linear4(x))
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
