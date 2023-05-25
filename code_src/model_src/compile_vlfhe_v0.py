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
sys.path.insert(1,"code_src/model_src/quantz/")
from train_v3_source_quantz import test_with_concrete
from cnn_v13_33bit_source_quantz import QTChessNET

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
#training_set, valid_set, test_set = np.split(wechess.sample(frac=1, random_state=42), [int(.6*len(wechess)), int(.8*len(wechess))])

# downsizing the training set size to avoid crash
training_set, valid_set, test_set = np.split(wechess.sample(frac=1, random_state=42), [int(.01*len(wechess)), int(.8*len(wechess))])
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

#model.load_state_dict(torch.load("code_src/model_src/quantz/source_quant3.pt",map_location = device)) #source

#model.load_state_dict(torch.load("server/model/target_clear.pt")) #target
model.pruning_conv(False)
# Test and accuracy
# test(model, test_loader, criterion)


cfg = Configuration(
        dump_artifacts_on_unexpected_failures=False,
        enable_unsafe_features=True,
        p_error=None,
        global_p_error=None)


accumlators = []
accum_bits = []

# # Prepare tests
list_inputs = []
list_targets = []


loop_trainset = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

for idx, (inputs_var, targets) in loop_trainset:

      data, target = inputs_var.clone().detach().float(), targets.clone().detach().float() #torch.tensor(inputs_var).float(), torch.tensor(targets).float() # 

      list_inputs.append(data)#.cpu().numpy())
      #list_targets.append(target)#.cpu().numpy())

loop_trainset.set_description(f"datasss [{idx}/{train_loader}]")

np_inputs = np.concatenate(list_inputs, axis=0)


# np_targets = np.concatenate(list_targets, axis=0)

#trainset = torch.tensor(trainset)
q_module_vl = compile_brevitas_qat_model(model, np_inputs, n_bits={"op_inputs":8, "op_weights":8}, use_virtual_lib=True, configuration=cfg, verbose_compilation=True)
# q_module_vl = compile_torch_model(model, np_inputs, import_qat=True ,use_virtual_lib=True, configuration=cfg, n_bits=8, verbose_compilation=True) #.detach()
print("flag1")
#accum_bits.append(q_module_vl.forward_fhe.graph.maximum_integer_bit_width())
print("flag2")
#print(accum_bits)

#accumlators.append(test_with_concrete(q_module_vl, testset, use_fhe=False, use_vl=False))

#print(accumlators)
