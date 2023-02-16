import torch
import torch.nn as nn
from torch import optim
import numpy as np 

"""
ASCII SET isometric1 http://asciiset.com/figletserver.html
"""
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

    valid_loss_min = np.Inf # track change in validation loss

    for e in range(epochs):

        train_loss = 0
        valid_loss = 0


        #      ___           ___           ___                       ___     
        #     /\  \         /\  \         /\  \          ___        /\__\    
        #     \:\  \       /::\  \       /::\  \        /\  \      /::|  |   
        #      \:\  \     /:/\:\  \     /:/\:\  \       \:\  \    /:|:|  |   
        #      /::\  \   /::\~\:\  \   /::\~\:\  \      /::\__\  /:/|:|  |__ 
        #     /:/\:\__\ /:/\:\ \:\__\ /:/\:\ \:\__\  __/:/\/__/ /:/ |:| /\__\
        #    /:/  \/__/ \/_|::\/:/  / \/__\:\/:/  / /\/:/  /    \/__|:|/:/  /
        #   /:/  /         |:|::/  /       \::/  /  \::/__/         |:/:/  / 
        #   \/__/          |:|\/__/        /:/  /    \:\__\         |::/  /  
        #                  |:|  |         /:/  /      \/__/         /:/  /   
        #                   \|__|         \/__/                     \/__/    


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



        #        ___           ___           ___                   ___           ___           ___           ___     
        #       /\__\         /\  \         /\__\      ___        /\  \         /\  \         /\  \         /\  \    
        #      /:/  /        /::\  \       /:/  /     /\  \      /::\  \       /::\  \        \:\  \       /::\  \   
        #     /:/  /        /:/\:\  \     /:/  /      \:\  \    /:/\:\  \     /:/\:\  \        \:\  \     /:/\:\  \  
        #    /:/__/  ___   /::\~\:\  \   /:/  /       /::\__\  /:/  \:\__\   /::\~\:\  \       /::\  \   /::\~\:\  \ 
        #    |:|  | /\__\ /:/\:\ \:\__\ /:/__/     __/:/\/__/ /:/__/ \:|__| /:/\:\ \:\__\     /:/\:\__\ /:/\:\ \:\__\
        #    |:|  |/:/  / \/__\:\/:/  / \:\  \    /\/:/  /    \:\  \ /:/  / \/__\:\/:/  /    /:/  \/__/ \:\~\:\ \/__/
        #    |:|__/:/  /       \::/  /   \:\  \   \::/__/      \:\  /:/  /       \::/  /    /:/  /       \:\ \:\__\  
        #     \::::/__/        /:/  /     \:\  \   \:\__\       \:\/:/  /        /:/  /     \/__/         \:\ \/__/  
        #      ~~~~           /:/  /       \:\__\   \/__/        \::/__/        /:/  /                     \:\__\    
        #                     \/__/         \/__/                 ~~            \/__/                       \/__/    

                    
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

    # avg loss
    train_loss = train_loss / len(trainloader.sampler)
    valid_loss = train_loss / len(validloader.sampler)

    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epochs, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min, valid_loss))

        torch.save(model.state_dict(), 'model_plain_chess.pt')
        valid_loss_min = valid_loss


#      ___           ___           ___           ___     
#     /\  \         /\  \         /\  \         /\  \    
#     \:\  \       /::\  \       /::\  \        \:\  \   
#      \:\  \     /:/\:\  \     /:/\ \  \        \:\  \  
#      /::\  \   /::\~\:\  \   _\:\~\ \  \       /::\  \ 
#     /:/\:\__\ /:/\:\ \:\__\ /\ \:\ \ \__\     /:/\:\__\
#    /:/  \/__/ \:\~\:\ \/__/ \:\ \:\ \/__/    /:/  \/__/
#   /:/  /       \:\ \:\__\    \:\ \:\__\     /:/  /     
#   \/__/         \:\ \/__/     \:\/:/  /     \/__/      
#                  \:\__\        \::/  /                 
#                   \/__/         \/__/            


def test(model, model_name, criterion):

    model.loard_state_dict(torch.load(model_name))

    test_loss = 0.0

    model.eval()

    for data, target in testloader:

        # forward
        output = model(data)
        # batch loss
        loss = criterion(output, target)
        # train_loss update
        test_loss += loss.item()*data.size(0)

        # TO DO comparison to true data
        # with conversion of output to wanted format


    # average test loss
    test_loss = test_loss/len(testloader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))
