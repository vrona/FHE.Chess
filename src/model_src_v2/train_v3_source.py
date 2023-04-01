import torch
import numpy as np
import wandb
from torch import optim
from tqdm import tqdm
import logging

"""
ASCII SET isometric1 http://asciiset.com/figletserver.html
"""
# logging.basicConfig(filename="std.log", 
# 					format='%(asctime)s %(message)s', 
# 					filemode='w')

# logger=logging.getLogger()
# logger.setLevel(logging.DEBUG)

# logger.debug("This is just a harmless debug message") 
# logger.info("This is just an information for you") 
# logger.warning("OOPS!!!Its a Warning") 
# logger.error("Have you try to divide a number by zero") 
# logger.critical("The Internet is not working....")

# CUDA's availability

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")

# if not train_on_gpu:
#     print('CPU Training... (CUDA not available)')

# else:
#     print('GPU Training...')

wandb.init(
        project = "Chess_App",

        config = {
        "learning_rate": 1.0e-3, #"weight_decay":0.099,
        "architecture": "CNN",
        "dataset": "White Black ELO 2000 arevel",
        "epochs": 5,
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
    optimizer = optim.Adam(model.parameters(), lr = wb_config.learning_rate) #weight_decay=wb_config.weight_decay
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

            data, target = data.to(torch.float).to(device), target.to(torch.float).to(device) #.to(torch.long) #
            #data, target = data.to(device), target.to(device)
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
            train_loss += loss.item()
            
            wandb.log({"loss": loss.item()})

            loop.set_description(f"Epoch_train [{epoch}/{n_epochs}]")
            loop.set_postfix(loss = loss.item())



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

            loss = criterion(output, target)

            # valid_loss update
            valid_loss += loss.item()
            
            wandb.log({"valid_loss": loss.item()})
            loop_valid.set_description(f"Epoch_valid [{epoch}/{n_epochs}]")
            loop_valid.set_postfix(validate_loss = loss.item())

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


def test(model, testloader, criterion):


    test_loss = 0.0
    accuracy = 0
    
    
    loop_test = tqdm(enumerate(testloader), total=len(testloader), leave=False)
    
    model.eval().to(device)
    
    with torch.no_grad():
        
        for batch_idx, (data, target) in loop_test:

            data, target = data.to(torch.float).to(device), target.to(torch.float).to(device)
            # forward
            output = model(data)

            # batch loss
            loss = criterion(output, target)

            # test_loss update
            test_loss += loss.item()

            # accuracy (output vs target)
            _, outdix = torch.max(output, 1)
     
            _, tardix = torch.max(target, 1)
     
            accuracy += (outdix == tardix).sum().item()

            wandb.log({"test_loss": loss.item(), "accuracy": 100 * accuracy / len(testloader)})
            loop_test.set_description(f"test [{batch_idx}/{len(testloader)}]")
            loop_test.set_postfix(testing_loss = loss.item())


        # average test loss
        test_loss = test_loss/len(testloader)
        print('Test Loss: {:.6f}\n'.format(test_loss))

        print('\nTest Accuracy: %2d%% ' % (100 * accuracy / len(testloader)))
    
    # closing the wandb logs
    wandb.finish()