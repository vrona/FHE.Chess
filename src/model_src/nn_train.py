import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np 

# CUDA's availability
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CPU Training... (CUDA not available)')

else:
    print('GPU Training...')

"""
LOADING SECTION
training_set = ZDataset(dataset['AN'])
training_loader =  DataLoader(training_set, batch_size=32, drop_last=True)
"""
trainloader = 0 
validloader = 0
testloader = 0



def validation(model, testloader, criterion):
    accuracy = 0
    test_loss = 0
    
    # get the x_data from testloader
    # get output from model and model.forward(input_data)
    # update test_loss += criterion(output, )
    # update accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy

def train(model, epochs=5):



    # loss function
    criterion = nn.CrossEntropyLoss() # To Change

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr = 0.01)

    
    for e in range(epochs):

        train_loss = 0
        valid_loss = 0

        ######################
        #_|_  ._   _.  o  ._  
        # |_  |   (_|  |  | | 
        ######################

        # model in training mode, dropout is on
        model.train()
        
        for data, target in trainloader:
            
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # clear the gradients from all variable
            optimizer.zero_grad()
            # forward
            output = model(data)
            # batch loss
            loss = criterion(output, target)
            # backward pass
            loss.backward()
            # single optimization step
            optimizer.step()
            # train_loss update
            train_loss += loss.item()*data.size(0)


        ##################################
        #     _.  |  o   _|   _.  _|_   _  
        #\/  (_|  |  |  (_|  (_|   |_  (/_ 
        ##################################
                    
        model.eval()

        for data, target in validloader:

            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # forward
            output = model(data)
            # batch loss
            loss = criterion(output, target)
            # train_loss update
            valid_loss += loss.item()*data.size(0)