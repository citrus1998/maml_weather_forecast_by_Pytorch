import csv
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class DataGenerator(object):
    def __init__(self, num_samples_per_class, batch_size, config={}):
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1
        self.batch_size = batch_size

        self.amp_range = config.get('amp_range', [-5.0, 35.0])
        self.phase_range = config.get('phase_range', [1, 12])

        self.dim_input = 1
        self.dim_output = 1


    def generate_weather_temperature_batch(self, train=True, input_idx=None):
        # set up input conditions
        csv_name = "data/tokyo_temp_191801-196712.csv"
        # csv_name = "data/tokyo_temp_196801-201712.csv"

        # load csv file
        temp_data = np.genfromtxt(csv_name, delimiter=',')

        # randomly select data from csv
        selected_data = np.array(random.sample(temp_data.tolist(), self.batch_size*self.num_samples_per_class*self.dim_input))
        label_batch, data_batch = np.transpose(selected_data)

        # add noise for 10%
        noise = np.random.uniform(-0.1, 0.1, self.batch_size*self.num_samples_per_class*self.dim_input)
        noisy_data_batch = data_batch + data_batch * noise

        # mock conditions
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])

        label_batch1 = np.reshape(label_batch, [self.batch_size, self.num_samples_per_class, self.dim_input])
        noisy_data_batch1 = np.reshape(noisy_data_batch, [self.batch_size, self.num_samples_per_class, self.dim_input])

        return label_batch1, noisy_data_batch1, amp, phase
    
    
    def generate_test_weather_temperature_batch(self, train=True, input_idx=None):
        # set up input conditions
        csv_name = "data/tokyo_temp_2018.csv"
        num_samples_per_class = 1
        batch_size = 12

        # load csv file
        temp_data = np.genfromtxt(csv_name, delimiter=',')

        # randomly select data from csv
        selected_data = np.array(random.sample(temp_data.tolist(), batch_size*num_samples_per_class*self.dim_input))
        label_batch, data_batch = np.transpose(selected_data)

        # add noise for 10%
        noise = np.random.uniform(-0.1, 0.1, batch_size*num_samples_per_class*self.dim_input)
        noisy_data_batch = data_batch + data_batch * noise

        # mock conditions
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])

        label_batch1 = np.reshape(label_batch, [batch_size, num_samples_per_class, self.dim_input])
        noisy_data_batch1 = np.reshape(noisy_data_batch, [batch_size, num_samples_per_class, self.dim_input])

        return label_batch1, noisy_data_batch1, amp, phase
