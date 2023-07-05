import sys
import argparse
import socket
import pickle
#from concrete.ml.deployment.fhe_client_server import FHEModelServer, FHEModelClient

sys.path.insert(1,"client/chess_env/")
from clone_chess import Clone_Chess


class Network:
    def __init__(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        parser = argparse.ArgumentParser(description='provide Server IP address')
        parser.add_argument('--server', type=str)
        parser.add_argument('--port', type=int)
        #self.parser.add_argument('-s', action= 'provide Server IP address')
        args = parser.parse_args()
        self.server = str(args.server) # 34.76.213.79 '0.0.0.0' #'127.0.0.1'
        self.port = int(args.port) # 3389

        if not self.server:
            print("The server's IP address is disfunctional.")
            raise SystemExit(1)

        #self.port = 3389        
        self.addr = (self.server, self.port)
        self.clone_chess = Clone_Chess()
        self.connect()
        #self.send(self.clone_chess.get_board())

    # NOT EFFICIENT TO ENHANCE
    def get_some_data(self):
        return self.clone_chess.get_board()


    # client connect to network, return loads data
    def connect(self):

        try:
            self.client.connect(self.addr)

        except socket.error as e:
            print(e)

    # client send data, return client receive data
    def send(self, data):

        try:
            self.client.send(pickle.dumps(data))
            prediction = pickle.loads(self.client.recv(2048*8))
            print(type(prediction))
            return prediction
        except socket.error as e:
            print(e)


# class OnDiskNetwork:
#     """Simulate a network on disk."""

#     def __init__(self):
#         # folder for server, client
#         self.server_dir = "server/model"
#         self.source_client = "client/source"
#         self.target_client = "client/target"

#         self.fhesource_client = FHEModelClient(path_dir=self.source_client, key_dir=self.source_client)
#         self.fhesource_client.load()


#     def run_fhe_inference(self, serialized_qx_new_encrypted, sub_model):
#         print("INSIDE FLAG 8.5")
#         self.fhemodel_server = FHEModelServer(self.server_dir + sub_model)
#         self.fhemodel_server.load()
#         print("INSIDE FLAG 9.5")
#         # with open(self.server_dir + sub_model + "/serialized_evaluation_keys.ekl", "rb") as f:
#         #     serialized_evaluation_keys = f.read()
#         serialized_evaluation_keys = self.fhesource_client.get_serialized_evaluation_keys()
#         print("FLAG 10.5")
#         serialized_result = self.fhemodel_server.run(serialized_qx_new_encrypted, serialized_evaluation_keys)

#         return serialized_result

#     def client_send_input_to_server_for_prediction(self, encrypted_input, sub_model):
#         """Send the input to the server and execute on the server in FHE."""
#         self.fhemodel_server = FHEModelServer(self.server_dir + sub_model)
#         self.fhemodel_server.load()
#         print("NETW client FLAG 1")
#         with open(self.server_dir + sub_model + "/serialized_evaluation_keys.ekl", "rb") as f:
#             serialized_evaluation_keys = f.read()
#         print("NETW client FLAG 2")
#         time_begin = time.time()
#         print("NETW client FLAG 3")

#         print("NETW client FLAG 3.5")
#         encrypted_prediction = self.fhemodel_server.run(encrypted_input, serialized_evaluation_keys)
#         print("NETW client FLAG 4")
#         time_end = time.time()
#         with open(self.server_dir + sub_model + "/encrypted_prediction.enc", "wb") as f:
#             f.write(encrypted_prediction)
#         print("NETW client FLAG 5")
#         return time_end - time_begin

#     def server_send_encrypted_prediction_to_client(self, sub_model):
#         """Send the encrypted prediction to the client."""
#         with open(self.server_dir + sub_model + "/encrypted_prediction.enc", "rb") as f:
#             encrypted_prediction = f.read()
#         return encrypted_prediction
