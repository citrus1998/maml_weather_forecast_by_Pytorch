import argparse
import math
import random
import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from collections import OrderedDict

import matplotlib as mpl
import matplotlib.pyplot as plt

from maml import MAML
from generate_dataset import DataGenerator


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=5) 
    parser.add_argument('--n_task_loop', type=int, default=250000) 
    parser.add_argument('--alpha', type=float, default=1e-2) 
    parser.add_argument('--epochs', type=int, default=1) 
    parser.add_argument('--update_batch_size', type=int, default=5)
    parser.add_argument('--meta_batch_size', type=int, default=25) 

    args = parser.parse_args() 

    datasets = DataGenerator(args.update_batch_size * 2, args.meta_batch_size)
    model = MAML()

    params = OrderedDict([
        ('input_net_weight', torch.Tensor(32, 1).uniform_(-1., 1.).requires_grad_()),
        ('input_net_bias', torch.Tensor(32).uniform_(-1., 1.).requires_grad_()),

        ('latent_net_weight', torch.Tensor(32, 32).uniform_(-1., 1.).requires_grad_()),
        ('latent_net_bias', torch.Tensor(32).uniform_(-1., 1.).requires_grad_()),

        ('output_net_weight', torch.Tensor(1, 32).uniform_(-1., 1.).requires_grad_()),
        ('output_net_bias', torch.Tensor(1).uniform_(-1., 1.).requires_grad_())
    ])

    optim = torch.optim.Adam(params.values(), lr=1e-3)

    for epoch in range(args.epochs):
        
        ''' Train Phase '''
        model.train()

        for itr in range(args.n_task_loop):
            
            ''' Support Set '''
            batch_x, batch_y, amp, phase = datasets.generate_weather_temperature_batch()

            num_classes = datasets.num_classes

            train_x = batch_x[:, :num_classes*args.update_batch_size, :]
            train_y = batch_y[:, :num_classes*args.update_batch_size, :]

            train_x = torch.from_numpy(train_x.astype(np.float32)).clone()
            train_y = torch.from_numpy(train_y.astype(np.float32)).clone()

            optim.zero_grad()

            new_params = params

            for k in range(args.K):
                
                pred_train_y = model(train_x, new_params)
                train_loss = F.l1_loss(pred_train_y, train_y)

                grads = torch.autograd.grad(train_loss, new_params.values(), create_graph=True)
                new_params = OrderedDict((name, param - args.alpha * grad) for ((name, param), grad) in zip(params.items(), grads))

                if itr % 100 == 0: 
                    print('Iteration %d -- Inner loop %d -- Loss: %.4f' % (itr, k, train_loss))

            val_x = batch_x[:, num_classes*args.update_batch_size:, :]
            val_y = batch_y[:, num_classes*args.update_batch_size:, :]

            val_x = torch.from_numpy(val_x.astype(np.float32)).clone()
            val_y = torch.from_numpy(val_y.astype(np.float32)).clone()

            pred_val_y = model(val_x, new_params)
            val_loss = F.l1_loss(pred_val_y, val_y)
            val_loss.backward(retain_graph=True)
            optim.step()

            if itr % 100 == 0: 
                print('Iteration %d -- Outer Loss: %.4f' % (itr, val_loss))

        ''' Test Phase '''
        model.eval()

        ''' Query Set '''
        batch_x, batch_y, amp, phase = datasets.generate_test_weather_temperature_batch(train=False)

        b = math.pi
        train_x = batch_x[:, :num_classes*args.update_batch_size, :]
        train_y = batch_y[:, :num_classes*args.update_batch_size, :]

        train_x = torch.from_numpy(train_x.astype(np.float32)).clone()
        train_y = torch.from_numpy(train_y.astype(np.float32)).clone()
            
        optim.zero_grad()
        
        new_params = params

        for k in range(args.K):
            pred_train_y = model(train_x, new_params)
            train_loss = F.l1_loss(pred_train_y, train_y)

            grads = torch.autograd.grad(train_loss, new_params.values(), create_graph=True)
            new_params = OrderedDict((name, param - args.alpha * grad) for ((name, param), grad) in zip(params.items(), grads))

        with torch.no_grad():
            test_x = batch_x[:, :num_classes*args.update_batch_size, :]
            test_y = batch_y[:, :num_classes*args.update_batch_size, :]
            
            test_x = torch.from_numpy(test_x.astype(np.float32)).clone()
            test_y = torch.from_numpy(test_y.astype(np.float32)).clone()
                
            test_pred_y = model(test_x, new_params)

            plt.plot(test_x.data.numpy(), test_y.data.numpy(), label='sin(x)')
            plt.plot(test_x.data.numpy(), test_pred_y.data.numpy(), label='net(x)')
            plt.plot(train_x.data.numpy(), train_y.data.numpy(), 'o', label='Examples')
            plt.legend()
                
            plt.savefig('maml-sine.png')



