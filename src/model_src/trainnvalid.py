import torch
import torch.nn as nn
from torch import optim
import numpy as np

"""
ASCII SET isometric1 http://asciiset.com/figletserver.html
"""
# CUDA's availability
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cuda")

# if not train_on_gpu:
#     print('CPU Training... (CUDA not available)')

# else:
#     print('GPU Training...')



def train_valid(model, trainloader, validloader, n_epochs=5):

    # loss function
    criterion = nn.CrossEntropyLoss()
    """
    initial_square = nn.CrossEntropyLoss()
    destination_square = nn.CrossEntropyLoss()

    2 prob distribution (which piece and which move)
    loss_initial_square = initial_square(output[:,0,:], y[:,0,:])
    loss_destination_square = destination_square(output[:,1,:], y[:,1,:])
    loss_move = loss_initial_square + loss_destination_square


    """
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr = 0.01)

    valid_loss_min = np.Inf # track change in validation loss

    for epoch in range(n_epochs):

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


        # model in training mode
        model.train()
        
        for data, target in trainloader:
            
            data, target = data.to(device), target.to(device)

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

            data, target = data.to(device), target.to(device)

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
        epoch, train_loss, valid_loss))
    
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


def test(model, model_name, testloader, criterion):

    model.loard_state_dict(torch.load(model_name))

    test_loss = 0.0

    model.eval()

    with torch.no_grad():

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
