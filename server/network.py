import sys
import time
import socket
import pickle
from concrete.ml.deployment.fhe_client_server import FHEModelServer

sys.path.insert(1,"client/chess_env")
from clone_chess import Clone_Chess


class Network:
    def __init__(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server = '0.0.0.0' #'127.0.0.1'
        self.port = 5555
        self.addr = (self.server, self.port)
        self.clone_chess = Clone_Chess()
        self.connect()

    # NOT EFFICIENT TO ENHANCE
    def get_some_data(self):
        return self.clone_chess.get_board()


    # client connect to network, return loads data
    def connect(self):
        try:
            self.client.connect(self.addr)
            return pickle.loads(self.client.recv(2048*3))
        except socket.error as e:
            print(e)

    # client send data, return client receive data
    def send(self, data):
        try:
            self.client.send(pickle.dumps(data))
            return pickle.loads(self.client.recv(2048*3))
        except socket.error as e:
            print(e)


class OnDiskNetwork:
    """Simulate a network on disk."""

    def __init__(self):
        # folder for server, client
        self.server_dir = "server"
        self.client_dir = "client"
        

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

    def server_send_encrypted_prediction_to_client(self, sub_model):
        """Send the encrypted prediction to the client."""
        with open(self.server_dir + sub_model + "/encrypted_prediction.enc", "rb") as f:
            encrypted_prediction = f.read()
        return encrypted_prediction
