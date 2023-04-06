import socket
import pickle
import sys

sys.path.insert(1,"server/model")
from inference_64bit import Inference


class Network:
    def __init__(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server = '127.0.0.1'
        self.port = 5555
        self.addr = (self.server, self.port)
        #self.prediction = self.connect()
        self.some_data = self.connect()
        self.inference = Inference()

    # def get_inference(self, input_board):
    #     return inference.predict(input_board)

    def get_data(self): return self.some_data 

    # client connect to network, return loads data
    def connect(self):
        try:
            self.client.connect(self.addr)
            return pickle.loads(self.client.recv(2048))
        except:
            pass

    # client send data, return client receive data
    def send(self, data):
        try:
            self.client.send(pickle.dumps(data))
            return pickle.loads(self.client.recv(2048))
        except socket.error as e:
            print(e)
