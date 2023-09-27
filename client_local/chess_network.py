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

        parser = argparse.ArgumentParser(description='provide Server IP_Address, port, local HvsH')
        parser.add_argument('--server', type=str, help="2 options: IP_Address for remote or local", required=True)
        parser.add_argument('--port', type=int, help="the remote server\'s port. default is 3389", default=3389)

        args = parser.parse_args()
        self.server = str(args.server)
        self.port = int(args.port) # 3389 Google Cloud Instance's firewall ok_port

        self.connected = False # helper variable when request if connected (see. input_ip() below, button.py,)

        if not self.server:
            print("the server's IP address is dysfunctional or local option not given.")
            raise SystemExit(1)

        if not self.port:
            print("default --port 3389, else replace with the desired port.")

        self.addr = (self.server, self.port)
        self.clone_chess = Clone_Chess()
        
        self.connect()

    def connect(self):
        """client connect to network"""
        try:
            self.client.connect(self.addr)
            self.connected = True
        except socket.error as e:
            self.connected = False
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

    def input_ip(self, connected):
        
        if not connected:
            print("\nTo enable AI mode, you must have your remote server running? yes/no")
            yes_no = str.casefold(input())

            if yes_no == "yes":
                print("Now, please provide the remote server ip_address:")
                ip_address = str(input())
                self.server = ip_address
                self.addr = (self.server, self.port)
                self.connect()
            
            if yes_no == "no":
                print("\nThen, please into your remote terminal run:\n$ python3 server/server_all.py -i (or --inference) \"clear\" or \"simfhe\" or \"deepfhe\"\nand wait terminal: \"socket is listening, server started\".")
                self.input_ip(connected)
            