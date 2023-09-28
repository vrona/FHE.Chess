import socket
import struct
import pickle
import argparse

class Network:
    def __init__(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        parser = argparse.ArgumentParser(description='provide Server IP_Address, port, local HvsH')
        parser.add_argument('--server', type=str, help="2 options: IP_Address for remote or local", required=True)
        parser.add_argument('--port', type=int, help="the remote server\'s port. default is 3389", default=3389)

        self.args = parser.parse_args()
        self.server = str(self.args.server)
        self.port = int(self.args.port) # 3389 Google Cloud Instance and AWS' ok firewall ports

        self.connected = False # helper variable when request if connected (see. input_ip() below, button.py,)

        if not self.server:
            print("the server's IP address is dysfunctional or local option not given.")
            raise SystemExit(1)

        if not self.port:
            print("default --port 3389, else replace with the desired port.")

        self.addr = (self.server, self.port)

        self.connect(self.addr)


    def connect(self, addr_sp):
        """client connect to network"""
        try:
            self.client.connect(addr_sp)
            self.connected = True
        except socket.error as e:
            if e.errno == 61: #"Connection refused":
                print("\n%s because your remote server may not running." % e)
                print("No worries: just choose between White AI and White Human modes and follow the requirements.")
            
            else:
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
            print("\nTo enable AI mode, you must have your remote server running. Does it? yes/no")
            yes_no = str.casefold(input())

            if yes_no == "yes":
                if self.server == "local":
                    print("Now, please provide the remote server ip_address:")
                    self.server = str(input())

                else:
                    self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.new_addr = (self.server, self.port)
                
                self.connect(self.new_addr)

            if yes_no == "no":
                print("\nThen, please into your remote terminal run:\n$ python3 server/server_all.py -i (or --inference) \"clear\" or \"simfhe\" or \"deepfhe\"\nand wait terminal: \"socket is listening, server started\".")
                self.input_ip(connected)