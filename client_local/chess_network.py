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

        parser = argparse.ArgumentParser(description='provide Server IP_Address and port')
        parser.add_argument('--server', type=str, help="the remote server's IP_Address is required")
        parser.add_argument('--port', type=int, help='the remote server\'s port', default=3389)

        args = parser.parse_args()
        self.server = str(args.server) # 34.76.213.79 0.0.0.0
        self.port = int(args.port) # 3389 Google Cloud Instance's firewall ok_port 

        if not self.server:
            print("The server's IP address is dysfunctional.")
            raise SystemExit(1)

  
        self.addr = (self.server, self.port)
        self.clone_chess = Clone_Chess()
        self.connect()


    def connect(self):
        """client connect to network"""
        try:
            self.client.connect(self.addr)
        except socket.error as e:
            print(e)


    def send(self, data):
        """client send data, return client receive data"""
        try:
            data_bytes= pickle.dumps(data)
            #print("SIZE OF SENT DATA:",len(data_bytes))
            self.client.sendall(struct.pack('I', len(data_bytes)))
            self.client.sendall(data_bytes)
            
            prediction = pickle.loads(self.client.recv(4096))
            return prediction
        except socket.error as e:
            print(e)

