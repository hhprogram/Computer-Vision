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
        # made the kernel size 4x4 as 224 is divisible by 4
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 32, 4)
        self.conv3 = nn.Conv2d(32, 32, 4)
        # kernel size of (2,2) and since left out stride it defaults to the size of the kernel. Note: putting a pooling layer after
        # every convolutional layer just like we did in Keras CV capstone
        self.maxPool1 = nn.MaxPool2d(2)
        self.maxPool2 = nn.MaxPool2d(2)
        self.maxPool3 = nn.MaxPool2d(2)

        self.drop1 = nn.Dropout2d(p=.3)
        self.drop2 = nn.Dropout2d(p=.15)
        self.drop3 = nn.Dropout2d(p=.10)
        self.drop4 = nn.Dropout2d(p=.10)


        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and layers to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # note: X starts of as the input image. And then just like we do in keras we keep using it as the output of each 
        # subsequent layer
        # get the number of examples within the input batch given X. (X is iterable of inputs. ie a variable IMAGES that has a bunch of 
        # images within it to be used in the network)
        batch_size = x.size()[0]
        x = self.maxPool1(F.relu(self.conv1(x)))
        x = self.maxPool2(F.relu(self.conv2(x)))
        x = self.maxPool3(F.relu(self.conv3(x)))
        x = self.drop1(x)
        # this should be like a Flatten layer. Where putting just the argument '-1' just takes the input layer dimensions and then 
        # just flattens it accordingly. (it works like numpy's 'reshape' method. batch_size is the first argument as we want it 
            # to have 10 'columns' / rows and then flat within each of those. We want that as to keep each individual input image
            # seperate)
        x = x.view(batch_size, -1)
        print(x.size())
        # note: this is how we make the 'Dense' (fully connected) layers like we did in Keras. There are called Linear here because remember
        # dense layer is a layer that is a linear operation of all the input vectors. We then will add on in the 'forward' function the 
        # activation function we want to use on this layer. Remember the last layer with the output of 136 values per input sample 
        # (note: per input sample means that we'll be outputting a 2-D output 10x136. Where 10 is just the batch_size - happens to be 10 in this
        # project, but if you change it in the juptyer notebook2 then it will change. And this 10 represents 10 images in each batch put into
        # this network. Therefore, the 136 part is the 136 'keypoints' one for x and one for y coordinate of each of the 68 keypoints.) 
        # will have no activation 
        # function because we aren't classifying but really more doing a regression. While the other ones we are classifying features within 
        # the face to help us determine facial keypoints
        # note: http://pytorch.org/docs/master/nn.html#linear.....the docs say the in_features is the value of the size of EACH input sample. This
        # is a key thing to not overlook. This is the size of the input layer when it's feeding forward and it only feeds one image at a time.
        # therefore this value is the dimension of each individual input sample and in our case x.size())[1] because x.size())[0] = batch size 
        # and x.size())[1] is the dimension of each individual image within the batch 
        # then NOTE: that the out_features parameter is also the size of EACH input sample output that we want. I.e what we want each layer to 
        # output for each individual layer. Therefore, after this first linear layer its output is 10x5000 
        self.linear1 = nn.Linear(list(x.size())[1], 5000)
        x = F.relu(self.linear1(x))
        self.linear2 = nn.Linear(list(x.size())[1], 2500)
        x = F.relu(self.linear2(x))
        self.linear3 = nn.Linear(list(x.size())[1], 500)
        x = F.relu(self.linear3(x))
        # now we output batch_sizex136 which is what we want.
        self.linear4 = nn.Linear(list(x.size())[1], 136)
        x = self.linear4(x)
        print(x.size())
        # x = F.relu(self.linear1(x))
        # x = self.drop2(x)
        # x = F.relu(self.linear2(x))
        # x = self.drop3(x)
        # x = F.relu(self.linear3(x))
        # x = self.drop4(x)
        # x = F.relu(self.linear4(x))
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
