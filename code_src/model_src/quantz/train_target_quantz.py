import torch
import numpy as np
import wandb
from torch import optim
from tqdm import tqdm

#torch.manual_seed(42)

"""
ASCII SET isometric1 http://asciiset.com/figletserver.html
"""


# CUDA's availability

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


wandb.init(
        project = "Chess_App",

        config = {
        "learning_rate": 1.0e-3,
        "architecture": "CNN",
        "dataset": "White Black ELO 2000 A.Revel kaggle dataset",
        "epochs": 5,
        }
    )

# copy config
wb_config = wandb.config

def train_valid(model, trainloader, validloader, criterion, n_epochs= wb_config.epochs):

    # loss function
    optimizer = optim.Adam(model.parameters(), lr = wb_config.learning_rate) #weight_decay=wb_config.weight_decay
    valid_loss_min = np.Inf # track change in training loss

    for epoch in range(n_epochs):

        train_loss = 0
        valid_loss = 0

        train_acc = 0
        valid_acc = 0

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
        
        for batch_idx, (chessboard, source, target) in loop:

            chessboard, source, target = chessboard.to(torch.float).to(device), source.to(torch.float).to(device), target.to(torch.float).to(device)

            # clear the gradients from all variable
            optimizer.zero_grad()

            # forward
            output = model(chessboard, source)

            # accuracy
            #print(output.shape,"||",target.shape)
            outdix = output.argmax(1)
            tardix = target.argmax(1)
            #train_acc += (outdix == tardix).sum().item()
            train_acc = (outdix == tardix).sum().item()
            #print(outdix.shape,"||",tardix.shape,"\n",train_acc)
            #monitor_train_acc = 100 * (train_acc /len(trainloader))
            monitor_train_acc = 100 * (train_acc /64)

            # batch loss
            loss = criterion(output, target)

            # backward pass
            loss.backward()
            # single optimization step
            optimizer.step()

            # train_loss update
            train_loss += loss.item()
            
            wandb.log({"loss": loss.item(), "train_accuracy" : monitor_train_acc})

            loop.set_description(f"Epoch_train [{epoch}/{n_epochs}]")
            loop.set_postfix(loss = loss.item(), train_accuracy = monitor_train_acc)



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

        for batch_idx, (chessboard, source, target) in loop_valid:

            chessboard, source, target = chessboard.to(torch.float).to(device), source.to(torch.float).to(device), target.to(torch.float).to(device)

            # forward
            output = model(chessboard, source)
            
            # accuracies
            outdix = output.argmax(1)
            tardix = target.argmax(1)
            valid_acc = (outdix == tardix).sum().item()
            
            monitoring_val_acc = 100 * valid_acc / 64 #len(validloader)

            # batch loss
            loss = criterion(output, target)

            # valid_loss update
            valid_loss += loss.item()

            wandb.log({"valid_loss": loss.item(), "valid_accuracy" : monitoring_val_acc})
            loop_valid.set_description(f"Epoch_valid [{epoch}/{n_epochs}]")
            loop_valid.set_postfix(validate_loss = loss.item(), valid_accuracy = monitoring_val_acc)

        # avg loss
        train_loss = train_loss / len(trainloader)
        valid_loss = valid_loss / len(validloader)

        train_acc = train_acc / len(trainloader)
        valid_acc = valid_acc / len(validloader)

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidate Loss: {:.6f}, \tTrain Acc: {:.6f}, \tValid Acc: {:.6f}'.format(
            epoch, train_loss, valid_loss, train_acc, valid_acc))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validate loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min, valid_loss))

            torch.save(model.state_dict(), "server/model/target_quant{}.pt".format(epoch))
            valid_loss_min = valid_loss

    model.pruning_conv(False)
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
        
        for batch_idx, (chessboard, source, target) in loop_test:

            chessboard, source, target = chessboard.to(torch.float).to(device), source.to(torch.float).to(device), target.to(torch.float).to(device)
            # forward
            output = model(chessboard, source)

             # batch loss
            loss = criterion(output, target)

            # test_loss update
            test_loss += loss.item()
            
            outdix = output.argmax(1)
            tardix = target.argmax(1)
     
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