import argparse
import math
import random
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from collections import OrderedDict

import matplotlib as mpl
import matplotlib.pyplot as plt

from maml2nd import MAML


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=5) 
    parser.add_argument('--n_task_loop', type=int, default=250000) 
    parser.add_argument('--alpha', type=float, default=1e-2) 
    parser.add_argument('--epochs', type=int, default=1) 

    args = parser.parse_args() 

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
            b = 0 if random.choice([True, False]) else math.pi
            train_x = torch.rand(4, 1) * 4 * math.pi - 2 * math.pi
            train_y = torch.sin(train_x + b)

            optim.zero_grad()

            new_params = params

            for k in range(args.K):
                
                pred_train_y = model(train_x, new_params)
                train_loss = F.l1_loss(pred_train_y, train_y)

                grads = torch.autograd.grad(train_loss, new_params.values(), create_graph=True)
                new_params = OrderedDict((name, param - args.alpha * grad) for ((name, param), grad) in zip(params.items(), grads))

                if itr % 100 == 0: 
                    print('Iteration %d -- Inner loop %d -- Loss: %.4f' % (itr, k, train_loss))

            val_x = torch.rand(4, 1) * 4 * math.pi - 2 * math.pi
            val_y = torch.sin(val_x + b)

            pred_val_y = model(val_x, new_params)
            val_loss = F.l1_loss(pred_val_y, val_y)
            val_loss.backward(retain_graph=True)
            optim.step()

            if itr % 100 == 0: 
                print('Iteration %d -- Outer Loss: %.4f' % (itr, val_loss))

        ''' Test Phase '''
        model.eval()

        ''' Query Set '''
        b = math.pi
        train_x = torch.rand(4, 1) * 4 * math.pi - 2 * math.pi
        train_y = torch.sin(train_x + b)
            
        optim.zero_grad()
        
        new_params = params

        for k in range(args.K):
            pred_train_y = model(train_x, new_params)
            train_loss = F.l1_loss(pred_train_y, train_y)

            grads = torch.autograd.grad(train_loss, new_params.values(), create_graph=True)
            new_params = OrderedDict((name, param - args.alpha * grad) for ((name, param), grad) in zip(params.items(), grads))

        with torch.no_grad():
            test_x = torch.arange(-2 * math.pi, 2 * math.pi, step=0.01).unsqueeze(1)
            test_y = torch.sin(test_x + b)
                
            test_pred_y = model(test_x, new_params)

            plt.plot(test_x.data.numpy(), test_y.data.numpy(), label='sin(x)')
            plt.plot(test_x.data.numpy(), test_pred_y.data.numpy(), label='net(x)')
            plt.plot(train_x.data.numpy(), train_y.data.numpy(), 'o', label='Examples')
            plt.legend()
                
            plt.savefig('maml-sine.png')



