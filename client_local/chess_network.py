import sys
import argparse
import socket
import struct
import pickle

sys.path.append("client/chess_env/")
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
            data_bytes= pickle.dumps(data)
            #print("SIZE OF SENT DATA:",len(data_bytes))
            self.client.sendall(struct.pack('I', len(data_bytes)))
            self.client.sendall(data_bytes)
            
            prediction = pickle.loads(self.client.recv(4096))
            return prediction
        except socket.error as e:
            print(e)

