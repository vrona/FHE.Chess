# Model Deployment

[client_server_fhe_deploy.py](../server_cloud/client_server_fhe_deploy.py) does the job.<br>

### Principle

Export the Source and Target models to their respective client-server locations (as below) and generate cryptographic elements:<br>

- **client (local machine)**:
    - chess code
    - network code.
- **server (remote machine)**:
    - client:
        - source: private keys generated for encryption (input_data: chessboard) and decryption (output: source square)
        - target: private keys generated for encryption (input_data_1: chessboard + input_data_2: source square) and decryption (output: target square)
    - server:
        - model:
            - source: public evaluation keys for inference on encrypted input_data
            - target: public evaluation keys for inference on encrypted input_data_1 + input_data_2

<br>


### Tool
The ```OnDiskNetwork``` class has been written by Zama's team and can be found in [deployment test cases](https://github.com/zama-ai/concrete-ml/tree/9096a9d4f106b486532ec77a26a2cb8e423ebcf1/tests/deployment).<br>

It literally copying the saved models and send it to client to generate private keys
```python
class OnDiskNetwork:
    """Simulate a network on disk."""

    def __init__(self):
        # Create 3 temporary folder for server, client and dev with tempfile
        self.server_dir = "server/model" # pylint: disable=consider-using-with
        self.client_dir = "client" # pylint: disable=consider-using-with
        self.dev_dir = "deploy" # pylint: disable=consider-using-with
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

```