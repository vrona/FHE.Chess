import time
import torch
from shutil import copyfile
from tempfile import TemporaryDirectory
import numpy

from concrete.ml.torch.compile import compile_brevitas_qat_model
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer

# quantized - source
from code_src.model_src.quantz.source_44cnn_quantz import QTChessNET

# quantized - target
from code_src.model_src.quantz.target_44cnn_quantz import QTtrgChessNET

# defining the processor
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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


# loading and compiling model
# quantized model 1 - aka source  
model_source = QTChessNET()

# quantized model 2 - aka target
model_target = QTtrgChessNET()

# loading zone
# quantized model 1 - aka source  
model_source.load_state_dict(torch.load("server/model/source_model_quant44.pt",map_location = device))
model_source.pruning_conv(False)

# quantized model 2 - aka target
model_target.load_state_dict(torch.load("server/model/target_model_quant44.pt",map_location = device))
model_target.pruning_conv(False)

## model 1
q_model_source = compile_brevitas_qat_model(model_source, train_input, n_bits={"model_inputs":4, "model_outputs":4})

## model 2
q_model_target = compile_brevitas_qat_model(model_target, train_input, n_bits={"model_inputs":4, "model_outputs":4})


# instantiating the network
network = OnDiskNetwork()

# Now that the model has been trained we want to save it to send it to a server
fhemodel_dev = FHEModelDev(network.dev_dir.name, model_dev)
fhemodel_dev.save()