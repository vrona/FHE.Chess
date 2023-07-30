import time
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import sys

from concrete.ml.torch.compile import compile_brevitas_qat_model
from data_compliance import get_train_input

## 1. DATASET
sys.path.append("model_src/")

from dataset_source import Chessset as source_set
from dataset_target import Chessset as target_set

# 2. MODEL AND TEST FUNC
# QUANTIZED #

sys.path.append("model_src/quantz/")
# source
from source_44cnn_quantz import QTChessNET

# quantized - target
from target_44cnn_quantz_eval import QTtrgChessNET

class CompileModel:
        def __init__(self):
            
             # dataset
            self.game_move_set = "data/wb_2000_300.csv"
            self.wechess = pd.read_csv(self.game_move_set)
            self.compiled_source = None
            self.compiled_target = None

            self.compile_models()

        def compile_models(self):
             ##
             # SPLITING DATA
             ##
             # split dataset splitted into: training_set (80%), valid_set (20%), test_set (20%)
             #training_set, valid_set, test_set = np.split(self.wechess.sample(frac=1, random_state=42), [int(.6*len(self.wechess)), int(.8*len(self.wechess))])

             # IMPORTANT downsizing the training set size to avoid crash causes by overload computation
             training_set, valid_set, test_set = np.split(self.wechess.sample(frac=1, random_state=42), [int(.002*len(self.wechess)), int(.8*len(self.wechess))])

             print(f"When compiling with concrete-ml, the size of training_set should be at least 100 data points, here: {len(training_set)}.")

             ##
             # DATA LOADING
             ##
             # Source
             # from dataset through Chessset class
             source_trainset = source_set(training_set['AN'], training_set.shape[0])

             # from Chessset class through torch Dataloader
             source_loader = DataLoader(source_trainset, batch_size = 64, shuffle=True, drop_last=True)
            
             # Target
             # from dataset through Chessset class
             target_trainset = target_set(training_set['AN'], training_set.shape[0])

             # from Chessset class through torch Dataloader
             target_loader = DataLoader(target_trainset, batch_size = 64, shuffle=True, drop_last=True)


             ###
             # MODELS INSTANTIATION SECTION
             ###

             # quantized model 1 - aka source  
             source_model = QTChessNET()

             # quantized model 2 - aka target
             target_model = QTtrgChessNET()


             # defining the processor
             device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

             # loading zone
             # quantized model 1 - aka source  
             source_model.load_state_dict(torch.load("weights/source_quantz.pt",map_location = device))
             source_model.pruning_conv(False)

             # quantized model 2 - aka target
             target_model.load_state_dict(torch.load("weights/target_quantz.pt",map_location = device))

             target_model.pruning_conv(False)


             # instantiate the train_loader (as array of tensor) as train_input (see. data_compliance.py)
             source_train_input = get_train_input(source_loader, target=False)

             target_train_input = get_train_input(target_loader, target=True)

        
             # COMPILATION SECTION
    
             print("Concrete-ml is compiling")

             start_compile = time.time()

             # source model compilation
             self.compiled_source = compile_brevitas_qat_model(source_model, source_train_input, n_bits={"model_inputs":4, "model_outputs":4})
             end_compile_1 = time.time()
             print(f"Source_M compilation finished in {end_compile_1 - start_compile:.2f} seconds")

             # target model compilation
             self.compiled_target = compile_brevitas_qat_model(target_model, target_train_input, n_bits={"model_inputs":4, "model_outputs":4})
             end_compile_2 = time.time()
             print(f"Target_M compilation finished in {end_compile_2 - start_compile:.2f} seconds")

             #Checking that the network is compatible with FHE constraints
             print("checking FHE constraints compatibility")

             bitwidth_source = self.compiled_source.fhe_circuit.graph.maximum_integer_bit_width()
             bitwidth_target = self.compiled_target.fhe_circuit.graph.maximum_integer_bit_width()
             print(
                f"Max bit-width: source {bitwidth_source} bits, target {bitwidth_target} bits" + " -> Fine in FHE!!"
                if bitwidth_source <= 16 and bitwidth_target <= 16
                else f"{bitwidth_source} or {bitwidth_target} bits too high for FHE computation"
                )


