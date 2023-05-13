import torch
from torch import optim
import numpy as np
import wandb
from tqdm import tqdm



torch.manual_seed(498846564)

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
    train_loss_min = np.Inf # track change in training loss

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

            data, target = data.to(torch.float).to(device), target.to(torch.float).to(device)

            # clear the gradients from all variable
            optimizer.zero_grad()

            # forward

            output = model(data)

            if batch_idx % 100 == 0:
             print(output)

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
        """
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
        """
        # avg loss
        train_loss = train_loss / len(trainloader)
        #valid_loss = valid_loss / len(validloader)

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, train_loss))#, valid_loss
        
        # save model if validation loss has decreased
        if train_loss <= train_loss_min:
            print('Training loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            train_loss_min, train_loss))

            torch.save(model.state_dict(), "source_model_quant_chess{}.pt".format(epoch))
            train_loss_min = train_loss
    
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
        
        for batch_idx, (data, target) in loop_test:

            data, target = data.to(torch.float32).to(device), target.to(torch.float32).to(device)
            # forward

            output = model(data)
            
            # batch loss
            loss = criterion(output, target)

            # test_loss update
            test_loss += loss.item()

            # accuracy (output vs target)
            outdix = output.argmax(1)
            tardix = target.argmax(1)

            accuracy += (outdix == tardix).sum().item()

            wandb.log({"test_loss": loss.item()})#, "accuracy": 100 * accuracy / len(testloader)})
            loop_test.set_description(f"test [{batch_idx}/{len(testloader)}]")
            loop_test.set_postfix(testing_loss = loss.item(), acc = accuracy)#)_rate = 100 * accuracy / len(testloader))


        # average test loss
        test_loss = test_loss/len(testloader)
        print('Test Loss: {:.6f}\n'.format(test_loss))
        print('\nTest Accuracy: %2d%% ' % (100 * accuracy / len(testloader)))


#       ___           ___                ___           ___           ___     
#      /\__\         /\__\              /\  \         /\__\         /\  \    
#     /:/  /        /:/  /             /::\  \       /:/  /        /::\  \   
#    /:/  /        /:/  /             /:/\:\  \     /:/__/        /:/\:\  \  
#   /:/__/  ___   /:/  /             /::\~\:\  \   /::\  \ ___   /::\~\:\  \ 
#   |:|  | /\__\ /:/__/             /:/\:\ \:\__\ /:/\:\  /\__\ /:/\:\ \:\__\
#   |:|  |/:/  / \:\  \             \/__\:\ \/__/ \/__\:\/:/  / \:\~\:\ \/__/
#   |:|__/:/  /   \:\  \                 \:\__\        \::/  /   \:\ \:\__\  
#    \::::/__/     \:\  \                 \/__/        /:/  /     \:\ \/__/  
#     ~~~~          \:\__\                            /:/  /       \:\__\    
#                    \/__/                            \/__/         \/__/  


def test_with_concrete(quantized_module, test_input,dtype_inputs=np.int64):
    """Test a neural network that is quantized and compiled with Concrete-ML."""

  
    print("beginning of test")
    # Casting the inputs into int64 is recommended
    # all_data = np.zeros((len(test_input)), dtype=dtype_inputs)
    # all_targets = np.zeros((len(test_input)), dtype=dtype_inputs)
    loop_vlfhe_test = tqdm(enumerate(test_input), total=len(test_input), leave=False)
    # Iterate over the test batches and accumulate predictions and ground truth labels in a vector

    accuracy = 0
    #for data, target in zip(test_input, test_target):
    for idx, (data, target) in loop_vlfhe_test:
      # from tensor to numpy
      data = data.cpu().detach().numpy()
      target = target.cpu().detach().numpy()
    
      # Quantize the inputs and cast to appropriate data type
      x_test_q = quantized_module.quantize_input(data).astype(dtype_inputs)

      # Accumulate the ground truth labels
      y_pred = quantized_module.quantized_forward(x_test_q, fhe="simulate")
      output = quantized_module.dequantize_output(y_pred)

      # Take the predicted class from the outputs and store it
      y_pred = np.argmax(output, 1)
      y_targ = np.argmax(target, 1)

      accuracy += np.sum(y_pred == y_targ)

      wandb.log({"accuracy": 100 * accuracy / len(test_input)})
      loop_vlfhe_test.set_description(f"test [{idx}/{len(test_input)}]")
      loop_vlfhe_test.set_postfix(acc = accuracy)#acc_rate = 100 * accuracy / len(testloader))

    # closing the wandb logs
    wandb.finish()