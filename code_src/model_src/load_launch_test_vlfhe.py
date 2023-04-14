import time
import tqdm
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import sys

from concrete.numpy.compilation.configuration import Configuration
from concrete.ml.torch.compile import compile_brevitas_qat_model
#from concrete.ml.quantization.quantized_module import QuantizedModule

from dataset_v3_source import Chessset
#from dataset_v3_target import Chessset

# clear - source
# sys.path.insert(1,"code_src/model_src/clear/")
# from train_v3_source_clear import test
# from cnn_v13_64bit_source_clear import PlainChessNET

# clear - target
# from train_v3_target import test
# from cnn_v13_64bit_target_unfhe import PlainChessNET

# quantized - source
sys.path.insert(1,"code_src/model_src/quantz/")
from train_v3_source_quantz import test
from cnn_v13_64bit_source_quantz import QTChessNET

# quantized - target
# from train_v3_target import test
# from cnn_v13_64bit_target_quantz import QTtrgChessNET


"""
LOADING SECTION
training_set = Chessset(dataset['AN'])
"""
#       ___           ___           ___                   ___                    ___           ___           ___           ___     
#      /\  \         /\  \         /\__\      ___        /\  \                  /\  \         /\  \         /\  \         /\  \    
#     /::\  \       /::\  \       /:/  /     /\  \       \:\  \                /::\  \       /::\  \        \:\  \       /::\  \   
#    /:/\ \  \     /:/\:\  \     /:/  /      \:\  \       \:\  \              /:/\:\  \     /:/\:\  \        \:\  \     /:/\:\  \  
#   _\:\~\ \  \   /::\~\:\  \   /:/  /       /::\__\      /::\  \            /:/  \:\__\   /::\~\:\  \       /::\  \   /::\~\:\  \ 
#  /\ \:\ \ \__\ /:/\:\ \:\__\ /:/__/     __/:/\/__/     /:/\:\__\          /:/__/ \:|__| /:/\:\ \:\__\     /:/\:\__\ /:/\:\ \:\__\
#  \:\ \:\ \/__/ \/__\:\/:/  / \:\  \    /\/:/  /       /:/  \/__/          \:\  \ /:/  / \/__\:\/:/  /    /:/  \/__/ \/__\:\/:/  /
#   \:\ \:\__\        \::/  /   \:\  \   \::/__/       /:/  /                \:\  /:/  /       \::/  /    /:/  /           \::/  / 
#    \:\/:/  /         \/__/     \:\  \   \:\__\       \/__/                  \:\/:/  /        /:/  /     \/__/            /:/  /  
#     \::/  /                     \:\__\   \/__/                               \::/__/        /:/  /                      /:/  /   
#      \/__/                       \/__/                                        ~~            \/__/                       \/__/    


game_move_set = "data/wb_2000_300.csv"
wechess = pd.read_csv(game_move_set)

# split dataset splitted into: training_set (80%), valid_set (20%), test_set (20%)
training_set, valid_set, test_set = np.split(wechess.sample(frac=1, random_state=42), [int(.6*len(wechess)), int(.8*len(wechess))])

#      ___           ___           ___           ___           ___       ___           ___           ___     
#     /\  \         /\  \         /\  \         /\  \         /\__\     /\  \         /\  \         /\  \    
#    /::\  \       /::\  \        \:\  \       /::\  \       /:/  /    /::\  \       /::\  \       /::\  \   
#   /:/\:\  \     /:/\:\  \        \:\  \     /:/\:\  \     /:/  /    /:/\:\  \     /:/\:\  \     /:/\:\  \  
#  /:/  \:\__\   /::\~\:\  \       /::\  \   /::\~\:\  \   /:/  /    /:/  \:\  \   /::\~\:\  \   /:/  \:\__\ 
# /:/__/ \:|__| /:/\:\ \:\__\     /:/\:\__\ /:/\:\ \:\__\ /:/__/    /:/__/ \:\__\ /:/\:\ \:\__\ /:/__/ \:|__|
# \:\  \ /:/  / \/__\:\/:/  /    /:/  \/__/ \/__\:\/:/  / \:\  \    \:\  \ /:/  / \/__\:\/:/  / \:\  \ /:/  /
#  \:\  /:/  /       \::/  /    /:/  /           \::/  /   \:\  \    \:\  /:/  /       \::/  /   \:\  /:/  / 
#   \:\/:/  /        /:/  /     \/__/            /:/  /     \:\  \    \:\/:/  /        /:/  /     \:\/:/  /  
#    \::/__/        /:/  /                      /:/  /       \:\__\    \::/  /        /:/  /       \::/__/   
#     ~~            \/__/                       \/__/         \/__/     \/__/         \/__/         ~~       

#datafromset = Chessset(wechess['AN'])
trainset = Chessset(training_set['AN'], training_set.shape[0])
validset = Chessset(valid_set['AN'], valid_set.shape[0])
testset = Chessset(test_set['AN'], test_set.shape[0])

train_loader = DataLoader(trainset, batch_size = 64, shuffle=True, drop_last=True)
valid_loader = DataLoader(validset, batch_size = 64, shuffle=True, drop_last=True)
test_loader = DataLoader(testset, batch_size = 1, shuffle=True, drop_last=True)


#model quantized - source  
model = QTChessNET()
#model quantized - target
#model = QTtrgChessNET()

#loss
criterion = nn.MSELoss()

## TRAINING
#train_valid(model, train_loader, valid_loader, criterion, criterion)

## TESTING
# defining the processor

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#model.load_state_dict(torch.load("server/model/source_clear.pt",map_location = device)) #source

model.load_state_dict(torch.load("code_src/model_src/quantz/source_model_quant_chess3.pt",map_location = device)) #source
#model.load_state_dict(torch.load("server/model/target_clear.pt")) #target
model.pruning_conv(False)
# Test and accuracy
# test(model, test_loader, criterion)

def test_with_concrete(quantized_module, test_loader, use_fhe, use_vl, dtype_inputs=np.int64):
    """Test a neural network that is quantized and compiled with Concrete-ML."""

    # Casting the inputs into int64 is recommended
    all_y_pred = np.zeros((len(test_loader)), dtype=dtype_inputs)
    all_targets = np.zeros((len(test_loader)), dtype=dtype_inputs)

    # Iterate over the test batches and accumulate predictions and ground truth labels in a vector
    idx = 0
    for data, target in tqdm(test_loader):
        data = data.numpy()

        # Quantize the inputs and cast to appropriate data type
        x_test_q = quantized_module.quantize_input(data).astype(dtype_inputs)

        # Accumulate the ground truth labels
        endidx = idx + target.shape[0]
        all_targets[idx:endidx] = target.numpy()

        # Iterate over single inputs
        for i in range(x_test_q.shape[0]):
            # Inputs must have size (N, C, H, W), we add the batch dimension with N=1
            x_q = np.expand_dims(x_test_q[i, :], 0)

            # Execute either in FHE (compiled or VL) or just in quantized
            if use_fhe or use_vl:
                out_fhe = quantized_module.forward_fhe.encrypt_run_decrypt(x_q)
                output = quantized_module.dequantize_output(out_fhe)
            else:
                output = quantized_module.forward_and_dequant(x_q)

            # Take the predicted class from the outputs and store it
            y_pred = np.argmax(output, 1)
            all_y_pred[idx] = y_pred
            idx += 1

    # Compute and report results
    n_correct = np.sum(all_targets == all_y_pred)
    return n_correct / len(test_loader)


"""def test_with_concrete(qmodule, testloader, use_fhe, use_vl):
    test_loss = 0.0
    accuracy = 0
    
    
    loop_test = tqdm(enumerate(testloader), total=len(testloader), leave=False)
    
    model.eval().to(device)
    
    with torch.no_grad():
        
        for batch_idx, (data, target) in loop_test:

            q_data = qmodule.quantize_input(data) #.astype(dtype_inputs) #data.to(torch.float32).to(device), target.to(torch.float32).to(device)
            # forward
            if use_fhe or use_vl:
                out_fhe = qmodule.forward_fhe.encrypt_run_decrypt(q_data)
                output = qmodule.dequantize_output(out_fhe)
            else:
                output = qmodule.forward_and_dequant(data)
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
"""

cfg = Configuration(
        dump_artifacts_on_unexpected_failures=False,
        enable_unsafe_features=True,
        p_error=None,
        global_p_error=None)


accumlators = []
accum_bits = []
q_module_vl = compile_brevitas_qat_model(model, train_loader, cfg, n_bits={"a_bits": 8, "w_bits":8},use_virtual_lib=True,configuration=cfg)

accum_bits.append(q_module_vl.forward_fhe.graph.maximum_integer_bit_width())


accumlators.append(
    test_with_concrete(
    q_module_vl, test_loader, use_fhe=False, use_vl=False
    )
)

