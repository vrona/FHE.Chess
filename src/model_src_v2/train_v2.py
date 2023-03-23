import torch
import numpy as np
import wandb
from torch import optim
from tqdm import tqdm
import logging

"""
ASCII SET isometric1 http://asciiset.com/figletserver.html
"""
logging.basicConfig(filename="std.log", 
					format='%(asctime)s %(message)s', 
					filemode='w')

logger=logging.getLogger()
logger.setLevel(logging.DEBUG)

logger.debug("This is just a harmless debug message") 
logger.info("This is just an information for you") 
logger.warning("OOPS!!!Its a Warning") 
logger.error("Have you try to divide a number by zero") 
logger.critical("The Internet is not working....")

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
        "learning_rate": 1.0e-3, #"weight_decay":0.099,
        "architecture": "CNN",
        "dataset": "White Black ELO 2000 arevel",
        "epochs": 10,
        }
    )

# copy config
wb_config = wandb.config

def train_valid(model, trainloader, validloader, criterion_f, criterion_t, n_epochs= wb_config.epochs):

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
            
            loss_from = criterion_f(output[:,0,:], target[:,0,:])
            loss_to = criterion_t(output[:,1,:], target[:,1,:])
            loss_0 = loss_from + loss_to

            #print("*****LOSS****",loss_0)
            #loss_1 = criterion(output[0,:], target[0,:])

            #print("loss :", loss)

            # backward pass
            loss_0.backward()
            # single optimization step
            optimizer.step()

            # train_loss update
            #train_loss += loss.item()*data.size(0)

            #print("train loss :",train_loss)
            
            wandb.log({"loss_0": loss_0.item()*data.size(0)}) #, "sum out": torch.sum(output)

            loop.set_description(f"Epoch_train [{epoch}/{n_epochs}]")
            loop.set_postfix(loss_0 = loss_0.item()*data.size(0)) #, out_sum = torch.sum(output)



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

            loss_from = criterion(output[:,0,:], target[:,0,:])
            loss_to = criterion(output[:,1,:], target[:,1,:])
            loss_0 = loss_from + loss_to
            # train_loss update
            valid_loss += loss_0.item()*data.size(0)
            
            wandb.log({"valid_loss": loss_0.item()*data.size(0)})
            loop_valid.set_description(f"Epoch_valid [{epoch}/{n_epochs}]")
            loop_valid.set_postfix(validate_loss = loss_0.item()*data.size(0))

        # avg loss
        train_loss = train_loss / len(trainloader) #.sampler
        valid_loss = train_loss / len(validloader) #.sampler

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
        
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
    accuracy = 0
    
    
    loop_test = tqdm(enumerate(testloader), total=len(testloader), leave=False)
    
    model.eval().to(device)
    
    with torch.no_grad():
        
        for batch_idx, (data, target) in loop_test:

            data, target = data.to(torch.float).to(device), target.to(torch.float).to(device)
            # forward
            output = model(data)
            # batch loss
            #loss = criterion(output, target)
            loss_from = criterion(output[:,0,:], target[:,0,:])
            loss_to = criterion(output[:,1,:], target[:,1,:])
            loss_0 = loss_from + loss_to

            # train_loss update
            test_loss += loss_0.item()*data.size(0)

            # WIP TO DO comparison to true data
            # with conversion of output to wanted format
            #prediction_item = output[-1].item()
   
            vals, indix = torch.max(output, 1)
 
            #indexes = output.data.max(1, keepdim=True)[1]
            #_, predtest = torch.max(output.data, 1)
            #print("pred",vals)
            #print(vals,'\n',indix)

            #print("pred", torch.max(vals),"index", torch.argmax(vals))

            # nummax = vals.detach().numpy()
            # indixofvals = indix.detach().numpy()

            # final_pred = np.argmax(nummax)

            # print(final_pred, indixofvals[final_pred])
            #prediction = output.data.max(1)
            accuracy += (indix == target).sum().item()

            wandb.log({"test_loss": loss_0.item()*data.size(0)})
            loop_test.set_description(f"test [{batch_idx}/{len(testloader)}]")
            loop_test.set_postfix(testing_loss = loss_0.item()*data.size(0))


        # average test loss
        test_loss = test_loss/len(testloader)
        print('Test Loss: {:.6f}\n'.format(test_loss))

        print('\nTest Accuracy: %2d%% ' % (100 * accuracy / len(testloader)))
    
    # closing the wandb logs
    wandb.finish()