# global package
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
from code_src.model_src.dataset_source import Chessset as Chessset_src
from code_src.model_src.dataset_target import Chessset as Chessset_trgt

## quantized model
### source
from code_src.model_src.quantz.source_44cnn_quantz import QTChessNET

### target
from code_src.model_src.quantz.target_44cnn_quantz import QTtrgChessNET

## data compliance for concrete-ml
from code_src.model_src.compile_fhe import get_train_input

# defining the processor
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""
GET TRAINING DATA SECTION
from data file to data compliance for concrete-ml
"""
# instantiation from data file
game_move_set = "data/wb_2000_300.csv"
wechess = pd.read_csv(game_move_set)

# split dataset to get only a small random fraction of training_set
## IMPORTANT downsizing the training set size to avoid crash causes by overload computation
training_set, valid_set, test_set = np.split(wechess.sample(frac=1, random_state=42), [int(.0005*len(wechess)), int(.8*len(wechess))])

print(f"When compiling with concrete-ml, tthe size of training_set should be at least 100 data points, here: {len(training_set)}.")

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

class OnDiskNetwork:
    """Simulate a network on disk."""

    def __init__(self):
        # Create 3 temporary folder for server, client and dev with tempfile
        self.server_dir = TemporaryDirectory()  # pylint: disable=consider-using-with
        self.client_dir = TemporaryDirectory()  # pylint: disable=consider-using-with
        self.dev_dir = TemporaryDirectory()  # pylint: disable=consider-using-with

    def client_send_evaluation_key_to_server(self, serialized_evaluation_keys):
        """Send the public key to the server."""
        with open(self.server_dir.name + "/serialized_evaluation_keys.ekl", "wb") as f:
            f.write(serialized_evaluation_keys)

    def client_send_input_to_server_for_prediction(self, encrypted_input):
        """Send the input to the server and execute on the server in FHE."""
        with open(self.server_dir.name + "/serialized_evaluation_keys.ekl", "rb") as f:
            serialized_evaluation_keys = f.read()
        time_begin = time.time()
        encrypted_prediction = FHEModelServer(self.server_dir.name).run(
            encrypted_input, serialized_evaluation_keys
        )
        time_end = time.time()
        with open(self.server_dir.name + "/encrypted_prediction.enc", "wb") as f:
            f.write(encrypted_prediction)
        return time_end - time_begin

    def dev_send_model_to_server(self):
        """Send the model to the server."""
        copyfile(self.dev_dir.name + "/server.zip", self.server_dir.name + "/server.zip")

    def server_send_encrypted_prediction_to_client(self):
        """Send the encrypted prediction to the client."""
        with open(self.server_dir.name + "/encrypted_prediction.enc", "rb") as f:
            encrypted_prediction = f.read()
        return encrypted_prediction

    def dev_send_clientspecs_and_modelspecs_to_client(self):
        """Send the clientspecs and evaluation key to the client."""
        copyfile(self.dev_dir.name + "/client.zip", self.client_dir.name + "/client.zip")

    def cleanup(self):
        """Clean up the temporary folders."""
        self.server_dir.cleanup()
        self.client_dir.cleanup()
        self.dev_dir.cleanup()

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
model_source.load_state_dict(torch.load("weights/source_model_quant44.pt",map_location = device))
model_source.pruning_conv(False)

# quantized model 2 - aka target
model_target.load_state_dict(torch.load("weights/target_model_quant44.pt",map_location = device))
model_target.pruning_conv(False)

"""
COMPILATION SECTION
get the quantized model
"""
## model 1
q_model_source = compile_brevitas_qat_model(model_source, train_input_src, n_bits={"model_inputs":4, "model_outputs":4})

## model 2
q_model_target = compile_brevitas_qat_model(model_target, train_input_trgt, n_bits={"model_inputs":4, "model_outputs":4})


"""
NETWORK/SAVING/SERVER SECTION
cf. zama documentation
"""
# instantiating the network
network = OnDiskNetwork()

# Now that the model has been trained we want to save it to send it to a server
fhemodel_src = FHEModelDev(network.dev_dir.name, q_model_source)
fhemodel_src.save()

fhemodel_trgt = FHEModelDev(network.dev_dir.name, q_model_target)
fhemodel_trgt.save()

# sending models to the server
network.dev_send_model_to_server()