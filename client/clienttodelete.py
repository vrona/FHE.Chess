import socket
import sys

# socket
server = socket.socket()

port = 5555
host = '127.0.0.1'

# connect
server.connect((host, port))

message = server.recv(1024)
print(message.decode("UTF-8"))

server.close()