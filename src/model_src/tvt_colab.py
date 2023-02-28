import torch
import numpy as np
import wandb
from torch import optim
from tqdm import tqdm

"""
ASCII SET isometric1 http://asciiset.com/figletserver.html
"""
# CUDA's availability

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")

# if not train_on_gpu:
#     print('CPU Training... (CUDA not available)')

# else:
#     print('GPU Training...')

wandb.init(
        project = "Chess App",

        config = {
        "learning_rate": 0.01,
        "architecture": "CNN",
        "dataset": "WhiteELO 2000 arevel",
        "epochs": 50,
        }
    )

# copy config
wb_config = wandb.config

def train_valid(model, trainloader, validloader, criterion, n_epochs= wb_config.epochs):

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

        #train_loss = 0
        #valid_loss = 0


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
        model.train().to(device)
        
        loop = tqdm(enumerate(trainloader), total=len(trainloader), leave=False)
        
        for batch_idx, (data, target) in loop:

          data, target = data.to(torch.float32).to(device), target.to(torch.float32).to(device)

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
          train_loss = loss.item()*data.size(0)

          wandb.log({"train_loss": train_loss})
          
          loop.set_description(f"Epoch [{epoch}/{n_epochs}]")
          loop.set_postfix(train_loss = loss.item()*data.size(0))
          #print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
          #print('Epoch: {} \tTraining Loss: {:.6f}\r'.format(epoch, train_loss))


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

        loop_valid = tqdm(enumerate(validloader), total=len(validloader),
        leave=False)

        model.eval().to(device)

        for batch_idx, (data, target) in loop_valid:

            data, target = data.to(torch.float32).to(device), target.to(torch.float32).to(device)

            # forward
            output = model(data)
            # batch loss
            loss = criterion(output, target)
            # train_loss update
            valid_loss = loss.item()*data.size(0)

            wandb.log({"valid_loss": valid_loss})
            loop_valid.set_description(f"Epoch [{epoch}/{n_epochs}]")
            loop_valid.set_postfix(valid_loss = loss.item()*data.size(0))
    # avg loss
    #train_loss = train_loss / len(trainloader.sampler)
    #valid_loss = train_loss / len(validloader.sampler)

    # print training/validation statistics 
    #print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        #epoch, train_loss, valid_loss))
          
    
    # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
              print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
              valid_loss_min, valid_loss))

              torch.save(model.state_dict(), "model_plain_chess.pt")
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


def test(model, testloader, criterion):


    test_loss = 0.0

    model.eval().to(device)

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
