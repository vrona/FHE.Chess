import os
import sys
import time
import torch
from torch.utils.data import DataLoader
from shutil import copyfile
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd

# zama packages
from concrete.ml.torch.compile import compile_brevitas_qat_model
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer

# internal class and method
## dataset management
from model_src.dataset_source import Chessset as Chessset_src
from model_src.dataset_target import Chessset as Chessset_trgt

## quantized model
### source
from model_src.quantz.source_44cnn_quantz import QTChessNET

### target
from model_src.quantz.target_44cnn_quantz_eval import QTtrgChessNET

## data compliance for concrete-ml
from server.data_compliance import get_train_input

# defining the processor
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class OnDiskNetwork:
    """Simulate a network on disk."""

    def __init__(self):
        # Create 3 temporary folder for server, client and dev with tempfile
        self.server_dir = "server/model" #TemporaryDirectory()  # pylint: disable=consider-using-with
        self.client_dir = "client" #TemporaryDirectory()  # pylint: disable=consider-using-with
        self.dev_dir = "deploy" #TemporaryDirectory()  # pylint: disable=consider-using-with
        self.empty_dev_dir()

    def empty_dev_dir(self):
        if len(os.listdir(self.dev_dir)):
            print("dev_dir is empty")
        else:
            print("dev_dir not empty")
    
    def client_send_evaluation_key_to_server(self, serialized_evaluation_keys, sub_model):
        """Send the public key to the server."""
        with open(self.server_dir + sub_model + "/serialized_evaluation_keys.ekl", "wb") as f:
            f.write(serialized_evaluation_keys)

    def client_send_input_to_server_for_prediction(self, encrypted_input, sub_model):
        """Send the input to the server and execute on the server in FHE."""
        with open(self.server_dir + sub_model + "/serialized_evaluation_keys.ekl", "rb") as f:
            serialized_evaluation_keys = f.read()
        time_begin = time.time()
        encrypted_prediction = FHEModelServer(self.server_dir).run(
            encrypted_input, serialized_evaluation_keys
        )
        time_end = time.time()
        with open(self.server_dir + sub_model + "/encrypted_prediction.enc", "wb") as f:
            f.write(encrypted_prediction)
        return time_end - time_begin

    def dev_send_model_to_server(self, sub_model):
        """Send the model to the server."""
        copyfile(self.dev_dir + sub_model + "/server.zip", self.server_dir + sub_model + "/server.zip")

    def server_send_encrypted_prediction_to_client(self, sub_model):
        """Send the encrypted prediction to the client."""
        with open(self.server_dir + sub_model + "/encrypted_prediction.enc", "rb") as f:
            encrypted_prediction = f.read()
        return encrypted_prediction

    def dev_send_clientspecs_and_modelspecs_to_client(self, sub_model):
        """Send the clientspecs and evaluation key to the client."""
        copyfile(self.dev_dir + sub_model + "/client.zip", self.client_dir + sub_model + "/client.zip")

    def cleanup(self):
        """Clean up the temporary folders."""
        self.server_dir.cleanup()
        self.client_dir.cleanup()
        self.dev_dir.cleanup()

"""
GET TRAINING DATA SECTION
from data file to data compliance for concrete-ml
"""
# instantiation from data file
game_move_set = "data/wb_2000_300.csv"
wechess = pd.read_csv(game_move_set)

# split dataset to get only a small random fraction of training_set
## IMPORTANT downsizing the training set size to avoid crash causes by overload computation
training_set = wechess.sample(frac=.002, random_state=42)

#training_set, valid_set, test_set = np.split(wechess.sample(frac=1, random_state=42), [int(.002*len(wechess)), int(.8*len(wechess))])

print(f"When compiling with concrete-ml, the size of training_set should be at least 100 data points, here: {len(training_set)}.")

# loading data
## from dataset through Chessset class
###source
trainset_src = Chessset_src(training_set['AN'], training_set.shape[0])

###target
trainset_trgt = Chessset_trgt(training_set['AN'], training_set.shape[0])

## from Chessset class through torch Dataloader
###source
train_src_loader = DataLoader(trainset_src, batch_size = 64, shuffle=True, drop_last=True)

###target
train_trgt_loader = DataLoader(trainset_trgt, batch_size = 64, shuffle=True, drop_last=True)

# get train_input data for compilation with concrete-ml
train_input_src = get_train_input(train_src_loader, target=False)    # source
train_input_trgt = get_train_input(train_trgt_loader, target=True)   # target


"""
LOADING MODEL SECTION
"""
# loading and compiling model
# quantized model 1 - aka source  
model_source = QTChessNET()

# quantized model 2 - aka target
model_target = QTtrgChessNET()

# loading zone
# quantized model 1 - aka source  
model_source.load_state_dict(torch.load("weights/source_quant7.pt",map_location = device))
model_source.pruning_conv(False)

# quantized model 2 - aka target
model_target.load_state_dict(torch.load("weights/target_4484_quant9.pt",map_location = device))
model_target.pruning_conv(False)

"""
COMPILATION SECTION
get the quantized model
"""
## model 1
q_model_source = compile_brevitas_qat_model(model_source, train_input_src, n_bits={"model_inputs":4, "model_outputs":4})

with open("mlir_source.txt", "w") as mlir:
    mlir.write(q_model_source.fhe_circuit.mlir)

## model 2
q_model_target = compile_brevitas_qat_model(model_target, train_input_trgt, n_bits={"model_inputs":4, "model_outputs":4})

with open("mlir_target.txt", "w") as mlir:
    mlir.write(q_model_target.fhe_circuit.mlir)
"""
NETWORK/SAVING/SERVER SECTION
cf. zama documentation
"""
# instantiating the network
network = OnDiskNetwork()

source_dev = network.dev_dir+"/source"
target_dev = network.dev_dir+"/target"

# saving trained-compiled model and sending to server
## model source
### Now that the model has been trained we want to save it to send it to a server
fhemodel_src = FHEModelDev(source_dev, q_model_source)
fhemodel_src.save()
print("model_source_saved")

### sending models to the server
network.dev_send_model_to_server("/source")
print("model_source_senttoserver")

## model target
fhemodel_trgt = FHEModelDev(target_dev, q_model_target)
fhemodel_trgt.save()
print("model_target_saved")

network.dev_send_model_to_server("/target")
print("model_target_senttoserver")

# send client specifications and evaluation key to the client
network.dev_send_clientspecs_and_modelspecs_to_client("/source")
network.dev_send_clientspecs_and_modelspecs_to_client("/target")


"""
CLIENT SECTION
cf. zama documentation
"""
source_client = network.client_dir+"/source"
target_client = network.client_dir+"/target"

#source
## client creation and loading the model
fhemodel_src_client = FHEModelClient(source_client, key_dir=source_client)

## private and evaluation keys creation
fhemodel_src_client.generate_private_and_evaluation_keys()

## get the serialized evaluation keys
serialz_eval_keys_src = fhemodel_src_client.get_serialized_evaluation_keys()
print(f"Evaluation 'source' keys size: {len(serialz_eval_keys_src) / (10**6):.2f} MB")

## send public key to server
network.client_send_evaluation_key_to_server(serialz_eval_keys_src, "/source")
print("sourceeval_key_senttoserver")

#target
## client creation and loading the model
fhemodel_trgt_client = FHEModelClient(target_client, key_dir=target_client)

## private and evaluation keys creation
fhemodel_trgt_client.generate_private_and_evaluation_keys()

## get the serialized evaluation keys
serialz_eval_keys_trgt = fhemodel_src_client.get_serialized_evaluation_keys()
print(f"Evaluation 'target' keys size: {len(serialz_eval_keys_trgt) / (10**6):.2f} MB")

## send public key to server
network.client_send_evaluation_key_to_server(serialz_eval_keys_trgt, "/target")
print("target_eval_key_senttoserver")
