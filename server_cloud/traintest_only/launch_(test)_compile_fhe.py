import time
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import sys

from concrete.ml.torch.compile import compile_brevitas_qat_model

from test_model_FHE import test_source_concrete, test_target_concrete

sys.path.append("server/")
from data_compliance import get_train_input

## 1. DATASET
sys.path.append("model_src/")

#source
from dataset_source import Chessset
#target
#from dataset_target import Chessset

# 2. MODEL AND TEST FUNC
# QUANTIZED #
sys.path.append("model_src/quantz/")

# source
from source_44cnn_quantz import QTChessNET

# quantized - target
#from target_44cnn_quantz_eval import QTtrgChessNET


"""
LOADING SECTION
training_set = Chessset(dataset['AN'])
"""

# 🅢🅟🅛🅘🅣 🅓🅐🅣🅐

game_move_set = "data/wb_2000_300.csv"
wechess = pd.read_csv(game_move_set)

# split dataset splitted into: training_set (60%), valid_set (20%), test_set (20%)
#training_set, valid_set, test_set = np.split(wechess.sample(frac=1, random_state=42), [int(.6*len(wechess)), int(.8*len(wechess))])

# IMPORTANT downsizing the training set size to avoid crash causes by overload computation
training_set, valid_set, test_set = np.split(wechess.sample(frac=1, random_state=42), [int(.001*len(wechess)), int(.8*len(wechess))])

print(f"When compiling with concrete-ml, tthe size of training_set should be at least 100 data points, here: {len(training_set)}.")


# 🅓🅐🅣🅐🅛🅞🅐🅓

# from dataset through Chessset class
trainset = Chessset(training_set['AN'], training_set.shape[0])
validset = Chessset(valid_set['AN'], valid_set.shape[0])
testset = Chessset(test_set['AN'], test_set.shape[0])

# from Chessset class through torch Dataloader
train_loader = DataLoader(trainset, batch_size = 64, shuffle=True, drop_last=True)
valid_loader = DataLoader(validset, batch_size = 64, shuffle=True, drop_last=True)
test_loader = DataLoader(testset, batch_size = 1, shuffle=True, drop_last=True)


# 🅜🅞🅓🅔🅛

# quantized model 1 - aka source  
model = QTChessNET()

# quantized model 2 - aka target
#model = QTtrgChessNET()

# 🅣🅔🅢🅣🅘🅝🅖 🅐🅝🅓 🅐🅒🅒🅤🅡🅐🅒🅨

# defining the processor
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# loading zone
# quantized model 1 - aka source  
model.load_state_dict(torch.load("weights/source_quantz.pt",map_location = device))

# quantized model 2 - aka target
#model.load_state_dict(torch.load("weights/target_quantz.pt",map_location = device))

model.pruning_conv(False)


# instantiate the train_loader (as array of tensor) as train_input
#source
train_input = get_train_input(train_loader)

#train
#train_input = get_train_input(train_loader, target=True)

#1 Compile to FHE
print("Concrete-ml is compiling")

start_compile = time.time()

q_module_vl = compile_brevitas_qat_model(model, train_input, n_bits={"model_inputs":4, "model_outputs":4})

end_compile = time.time()

print(f"Compilation finished in {end_compile - start_compile:.2f} seconds")

#2 Check that the network is compatible with FHE constraints
print("checking FHE constraints compatibility")

bitwidth = q_module_vl.fhe_circuit.graph.maximum_integer_bit_width()
print(
    f"Max bit-width: {bitwidth} bits" + " -> it works in FHE!!"
    if bitwidth <= 16
    else f"{bitwidth} bits too high for FHE computation"
)

#3 test concrete and monitoring accuracy
print("Test with concrete")
start_time_encrypt = time.time()

#source
test_source_concrete(q_module_vl,test_loader)

#target
#test_target_concrete(q_module_vl,test_loader)

print("Time per inference under FHE context:", (time.time()-start_time_encrypt/len(test_loader)))
