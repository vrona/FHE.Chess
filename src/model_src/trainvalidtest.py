import torch
import numpy as np
import wandb
from torch import optim
from tqdm import tqdm

"""
ASCII SET isometric1 http://asciiset.com/figletserver.html
"""
# CUDA's availability

#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

# if not train_on_gpu:
#     print('CPU Training... (CUDA not available)')

# else:
#     print('GPU Training...')

wandb.init(
        project = "Chess_App",

        config = {
        "learning_rate": 0.0018,
        "architecture": "CNN",
        "dataset": "WhiteELO 2000 arevel",
        "epochs": 10,
        }
    )

# copy config
wb_config = wandb.config

def train_valid(model, trainloader, validloader, criterion_from, criterion_to, n_epochs= wb_config.epochs):

    # loss function
    
    """
    initial_square = nn.CrossEntropyLoss()
    destination_square = nn.CrossEntropyLoss()

    2 prob distribution (which piece and which move)
    loss_initial_square = initial_square(output[:,0,:], y[:,0,:])
    loss_destination_square = destination_square(output[:,1,:], y[:,1,:])
    loss_move = loss_initial_square + loss_destination_square


    """
    optimizer = optim.Adam(model.parameters(), lr = wb_config.learning_rate)
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

        loop = tqdm(enumerate(trainloader), total=len(trainloader), leave=False)

        # model in training mode
        model.train().to(device)
        
        for batch_idx, (data, target) in loop:
            
            data, target = data.to(torch.float).to(device), target.to(torch.long).to(device)

            # clear the gradients from all variable
            optimizer.zero_grad()
            # forward
            output = model(data)

            # batch loss

            # print("Out sum:",torch.sum(output[:,0,:]), "Target sum:",torch.sum(target[:,0,:]))
            print(output[:,0,:].shape, target[:,0,:].shape)
            loss = criterion_from(output[:,0,:], target[:,0,:])
            ##loss_from = criterion_from(output[:,0,:], target[:,0,:])

            #print("loss_from :",loss_from)
            ##loss_to= criterion_to(output[:,1,:], target[:,1,:])
            #print("loss_to :", loss_to)
            ##loss = loss_from + loss_to
            #print("loss :", loss)

            # backward pass
            loss.backward()
            # single optimization step
            optimizer.step()

            # train_loss update
            train_loss += loss.item()*data.size(0)

            #print("train loss :",train_loss)

            wandb.log({"train_loss": loss.item()*data.size(0)})

            loop.set_description(f"Epoch [{epoch}/{n_epochs}]")
            loop.set_postfix(from_loss = loss.item()*data.size(0))



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

        loop_valid = tqdm(enumerate(validloader), total=len(validloader), leave=False)                    

        model.eval().to(device)

        for batch_idx, (data, target) in loop_valid:

            data, target = data.to(torch.float).to(device), target.to(torch.float).to(device)

            # forward
            output = model(data)
            # batch loss
            loss_from = criterion_from(output[:,0,:], target[:,0,:])
            loss_to= criterion_to(output[:,1,:], target[:,1,:])
            loss = loss_from + loss_to
            # train_loss update
            valid_loss += loss.item()*data.size(0)
            
            wandb.log({"valid_loss": loss.item()*data.size(0)})
            loop_valid.set_description(f"Epoch [{epoch}/{n_epochs}]")
            loop_valid.set_postfix(valid_loss = loss.item()*data.size(0))

        # avg loss
        train_loss = train_loss / len(trainloader) #.sampler
        valid_loss = valid_loss / len(validloader) #.sampler

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min, valid_loss))

        torch.save(model.state_dict(), "model_plain_chess.pt")
        valid_loss_min = valid_loss

    # closing the wandb logs
    wandb.finish()
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


def test(model, testloader, criterion_from, criterion_to):


    test_loss = 0.0

    model.eval().to(device)

    with torch.no_grad():

        for data, target in testloader:

            # forward
            output = model(data)
            
            # batch loss
            #loss = criterion_from(output, target)
            loss_from = criterion_from(output[:,0,:], target[:,0,:])
            loss_to= criterion_to(output[:,1,:], target[:,1,:])
            loss = loss_from + loss_to
            
            # train_loss update
            test_loss += loss.item()*data.size(0)

            # TO DO comparison to true data
            # with conversion of output to wanted format


        # average test loss
        test_loss = test_loss/len(testloader)
        print('Test Loss: {:.6f}\n'.format(test_loss))
