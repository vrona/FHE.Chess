# global package
import os
import time
from shutil import copyfile

# zama packages
from concrete.ml.deployment import FHEModelClient, FHEModelServer

class OnDiskNetwork:
    """Simulate a network on disk."""

    def __init__(self):
        # Create 3 temporary folder for server, client and dev with tempfile
        self.server_dir = "server/model" #TemporaryDirectory()  # pylint: disable=consider-using-with
        self.client_dir = "client" #TemporaryDirectory()  # pylint: disable=consider-using-with
        self.dev_dir = "code_src/deploy" #TemporaryDirectory()  # pylint: disable=consider-using-with
        # self.empty_dev_dir()

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
        #self.server_dir.cleanup()
        #self.client_dir.cleanup()
        self.dev_dir.cleanup()


"""
CLIENT SECTION
cf. zama documentation
"""
# instantiating the network
network = OnDiskNetwork()

source_client = network.client_dir+"/source"
target_client = network.client_dir+"/target"

# #source
# ## client creation and loading the model
# fhemodel_src_client = FHEModelClient(source_client, key_dir=source_client)
# print("flag_source_loaded")
# ## private and evaluation keys creation
# fhemodel_src_client.generate_private_and_evaluation_keys()
# print("flag_source_keys")

# ## get the serialized evaluation keys
# serialz_eval_keys_src = fhemodel_src_client.get_serialized_evaluation_keys()
# print(f"Evaluation 'source' keys size: {len(serialz_eval_keys_src) / (10**6):.2f} MB")

# network.client_send_evaluation_key_to_server(serialz_eval_keys_src, "/source")
# print("flat_eval_key_sent")


#target
## client creation and loading the model
fhemodel_trgt_client = FHEModelClient(target_client, key_dir=target_client)
print("flag_target_loaded")
## private and evaluation keys creation
fhemodel_trgt_client.generate_private_and_evaluation_keys()
print("flag_target_keys")

## get the serialized evaluation keys
serialz_eval_keys_trgt = fhemodel_trgt_client.get_serialized_evaluation_keys()
print(f"Evaluation 'target' keys size: {len(serialz_eval_keys_trgt) / (10**6):.2f} MB")

network.client_send_evaluation_key_to_server(serialz_eval_keys_trgt, "/target")
print("flat_eval_key_sent")
