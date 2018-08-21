

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import math
import h5py

import os
from skimage import io, transform
from tqdm import trange, tqdm

class Data:

    def __init__(self, 
            percent_load = 0.02,
            percent_training = 0.8,
            useNormalization = True,
            filepath = '../mri-data/Cardic_Undersampled_for_CS/training_data_2.h5'
        ): 

        self.percent_load = percent_load
        self.percent_training = percent_training
        self.useNormalization = useNormalization
        self.filepath = filepath
        
        print("Loading data ....")
        self.x_input, self.y_input = self.load()
        self.generate()
        print("Loading and data formating complete!")
        print(self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape)

    def get_image(self, index):
        return np.array(self.h5.get(self.keys[index]))

    def load(self):

        # Load h5 data
        self.h5 = h5py.File(self.filepath, 'r')
        self.keys = self.h5.keys()
        self.keys = list(self.keys)

        # Show data format for h5 data
        print("Data format: ", self.get_image(0))    

        # Load and format data in to array of shape -> (Count, Width, Height, Channels)
        keyCount = len(self.keys)
        maxIndex = math.floor(keyCount * self.percent_load);
        for i in tqdm(range(1, maxIndex + 1)):
            if(i == 1):
                input = self.get_image(i)[None, :]
            else:
                input = np.concatenate( (input, self.get_image(i)[None, :]), axis=0)
        input = np.transpose(input, (0, 2, 3, 1))

        # Grab Correct Indices
        x_input = input[:, :, :, 4] # Load Complex Undersampled Image
        x_input = np.concatenate(( np.real(x_input)[:, :, :, None], np.imag(x_input)[:, :, :, None] ), axis=3)
        y_input = np.real(input[:, :, :, 0:2]) # Load Real Order, Imaginary Order
        return x_input, y_input 

    def generate(self):
        if self.useNormalization:
            self.x_input, self.x_min, self.x_max = self.normalize(self.x_input)
            self.y_input, self.y_min, self.y_max = self.normalize(self.y_input)

        print('Reordering Values: Max: ' + str(self.y_max) + ' Min:' + str(self.y_min))

        index = int(self.percent_training * self.x_input.shape[0]) # Split index 
        self.x_train = self.x_input[0:index, :]
        self.x_test = self.x_input[index:, :]
        self.y_train = self.y_input[0:index, :]
        self.y_test = self.y_input[index:, :]

    def normalize(self, data):  
        max = np.max(data)
        min = np.min(data)
        return (data - min) / (max - min), min, max

    def denormalize(self, data, min, max):
        return data * (max - min) + min

    def next_batch(self, batch_size):
        length = self.x_train.shape[0]
        indices = np.random.randint(0, length, batch_size) # Grab batch_size values randomly
        return [self.x_train[indices], self.y_train[indices]]

    def plot(self, index): 
        plt.subplot(2,2,1) 
        plt.imshow(self.x_train[index, :, :, 0])
        plt.subplot(2,2,2)
        plt.imshow(self.x_train[index, :, :, 1])
        plt.subplot(2,2,3)
        plt.imshow(self.y_train[index, :, :, 0])
        plt.subplot(2,2,4)
        plt.imshow(self.y_train[index, :, :, 1])
        plt.show() 

if __name__ == '__main__':
    data = Data()
    data.plot(1) 
