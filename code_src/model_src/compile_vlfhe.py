import time
import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm

from concrete.numpy.compilation.configuration import Configuration
from concrete.ml.torch.compile import compile_brevitas_qat_model, compile_torch_model
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
#sys.path.insert(1,"code_src/model_src/quantz/")
from train_v3_source_quantz_FHE import test_with_concrete
from cnn_v13_33bit_source_quantz import QTChessNET
#from cnn_v13_88bit_source_quantz import QTChessNET

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


game_move_set = "/content/gdrive/MyDrive/FHX/wb_2000_300.csv"
wechess = pd.read_csv(game_move_set)

# split dataset splitted into: training_set (80%), valid_set (20%), test_set (20%)
#training_set, valid_set, test_set = np.split(wechess.sample(frac=1, random_state=42), [int(.6*len(wechess)), int(.8*len(wechess))])

# downsizing the training set size to avoid crash
training_set, valid_set, test_set = np.split(wechess.sample(frac=1, random_state=42), [int(.005*len(wechess)), int(.8*len(wechess))])
print(len(training_set), len(valid_set), len(test_set))

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
#device = torch.device("cpu")
#model.load_state_dict(torch.load("server/model/source_clear.pt",map_location = device)) #source

model.load_state_dict(torch.load("/content/gdrive/MyDrive/FHX/weigths/backup_CNN13_source_quantz_33bits_64 prune/source_quant3.pt",map_location = device)) #source
#model.load_state_dict(torch.load("/content/gdrive/MyDrive/FHX/weigths/backup_CNN13_source_quantz_88bits_48 prune BEST/source_model_quant_chess3.pt",map_location = device)) #source

#model.load_state_dict(torch.load("server/model/target_clear.pt")) #target
model.pruning_conv(False)
# Test and accuracy
# test(model, test_loader, criterion)


cfg = Configuration(
        dump_artifacts_on_unexpected_failures=False,
        enable_unsafe_features=True,
        p_error=None,
        global_p_error=None)


## Prepare tests
list_train_inputs = []
list_test_inputs = []
list_test_targets = []

## preparation train input data
loop_trainset = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

for idx, (inputs_var, targets) in loop_trainset:
      data, target = inputs_var.clone().detach().float(), targets.clone().detach().float() #torch.tensor(inputs_var).float(), torch.tensor(targets).float() # 
      list_train_inputs.append(data)

loop_trainset.set_description(f"datasss [{idx}/{train_loader}]")
train_input = np.concatenate(list_train_inputs, axis=0)

## preparation test input data
loop_testset = tqdm(enumerate(test_loader), total=len(test_loader), leave=False)

for idx, (inputs_test, targets) in loop_testset:
      data, target = inputs_test.clone().detach().float(), targets.clone().detach().float()
      list_test_inputs.append(data)
      list_test_targets.append(target)
loop_testset.set_description(f"datasss [{idx}/{test_loader}]")

test_input = np.concatenate(list_test_inputs, axis=0)
test_targets = np.concatenate(list_test_targets, axis=0)

## compilation to FHE model
def compile_and_test(custom_model, trainset, testset, target, use_sim=True):
  # Compile the model
    print("Compiling the model")
    start_compile = time.time()
    quantized_numpy_module = compile_brevitas_qat_model(
        custom_model,  # Our model
        trainset,  # A representative input-set to be used for both quantization and compilation
        n_bits={"op_inputs":3, "op_weights":3},
    )
    end_compile = time.time()
    print(f"Compilation finished in {end_compile - start_compile:.2f} seconds")

    # Check that the network is compatible with FHE constraints
    bitwidth = quantized_numpy_module.fhe_circuit.graph.maximum_integer_bit_width()
    print(
        f"Max bit-width: {bitwidth} bits" + " -> it works in FHE!!"
        if bitwidth <= 16
        else " too high for FHE computation"
    )

    # Execute prediction using simulation
    # (not encrypted but fast, and results are equivalent)

    if not use_sim:
        print("Generating key")
        start_key = time.time()
        quantized_numpy_module.fhe_circuit.keygen()
        end_key = time.time()
        print(f"Key generation finished in {end_key - start_key:.2f} seconds")

    fhe_mode = "simulate" if use_sim else "execute"

    predictions = numpy.zeros_like(target)

    print("Starting inference")
    start_infer = time.time()
    predictions = quantized_numpy_module.forward(testset, fhe=fhe_mode).argmax(1)
    end_infer = time.time()

    print(f"Compilation finished in {end_compile - start_compile:.2f} seconds")
    if not use_sim:
        print(f"Key generation finished in {end_key - start_key:.2f} seconds")
        print(
            f"Inferences finished in {end_infer - start_infer:.2f} seconds "
            f"({(end_infer - start_infer)/len(testset):.2f} seconds/sample)"
        )

    # Compute accuracy
    accuracy = numpy.mean(predictions == target) * 100
    print(f"Test Quantized Accuracy: {accuracy:.2f}% on {len(testset)} examples.")
    return bitwidth, accuracy, predictions, quantized_numpy_module

compile_and_test(model,train_input, test_input, test_targets, use_sim=True)
"""#1 Compile to FHE
print("Step 1: compilation")
q_module_vl = compile_brevitas_qat_model(model, train_input, n_bits={"op_inputs":3, "op_weights":3}, use_virtual_lib=True, configuration=cfg, verbose_compilation=False)


# checking that network compatibility with FHE limits
bitwidth = q_module_vl.forward_fhe.graph.maximum_integer_bit_width()
print(f"Max bit-width: {bitwidth} bits,"+" the network is compatible!"
      if bitwidth <= 16
      else "bitwidth too high for FHE computation")

#2 Generate Keys
print("Step 2: Keygen")
start_time_keys = time.time()
#q_module_vl.forward_fhe.keygen()
# alternative
q_module_vl.keygen()
print(f"Keygen time: {time.time()-start_time_keys:.2f}s")

#3 Execute in FHE on encrypted data
print("Step 3: FHE Execution")
start_time_encrypt = time.time()

test_with_concrete(q_module_vl, test_loader, use_fhe=True, use_vl=False)
print("Time per inference under FHE context:", (time.time()-start_time_encrypt/len(test_loader)))"""