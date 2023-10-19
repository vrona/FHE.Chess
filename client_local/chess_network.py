import socket
import ipaddress
import struct
import pickle
import argparse

class Network:
    def __init__(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        parser = argparse.ArgumentParser(description='provide DevMode, Server IP_Address, port')
        parser.add_argument('--devmode', type=str, help="bool. if True gives access to HvsH and AIvsAI", choices=("True","False"))#,default=False)
        parser.add_argument('--server', type=str, help="IP_Address for remote", required=True)
        parser.add_argument('--port', type=int, help="the remote server\'s port. default is 3389", default=3389)

        self.args = parser.parse_args()
        self.devmode = True if self.args.devmode == "True" else False # developer mode which enable testing the Human vs Human or AIvsAI

        self.is_ip_address_valid(self.args.server) # checking that IP_Address is  IPv4 or IPv6.

        self.port = int(self.args.port) # 3389 Google Cloud Instance and AWS' ok firewall ports

        self.connected = False # helper variable when request if connected (see. input_ip() below, button.py,)

        if not self.port:
            print("default --port 3389, else replace with the desired port.")

        self.addr = (self.server, self.port)

        self.connect(self.addr)


    def is_ip_address_valid(self, ip_string):
        """checking IPv4 of IPv6 compliances"""
        try:
            ip_addr = ipaddress.ip_address(ip_string)
            print("%s is a valid ip address" % ip_addr)
            self.server = ip_string
        except ValueError:
            print("\nOoops, the IP_Address has been mistyped. Double check it and just paste it below please:")
            new_ip_string = str(input())
            self.is_ip_address_valid(new_ip_string)

    def connect(self, addr_sp):
        """client connect to network"""
        try:
            self.client.connect(addr_sp)
            self.connected = True
        except socket.error as e:
            if e.errno == 61: #"Connection refused":
                print("\n%s because your remote server may not running." % e)
                self.reconnect(self.connected)
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


    def reconnect(self, connected):

        if not connected:
            print("\nDoes your remote server run? yes/no")
            yes_no = str.casefold(input())

            if yes_no == "yes":
                self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.new_addr = (self.server, self.port)
                self.connect(self.new_addr)

            if yes_no == "no":
                print("\nThen, please into your remote terminal run:\n$ python3 server/server_all.py -i (or --inference) \"clear\" or \"simfhe\" or \"deepfhe\"\nand wait terminal: \"socket is listening, server started\".")
                self.reconnect(connected)