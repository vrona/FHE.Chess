import socket
from _thread import *
import sys
#from main import Main

server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# main = Main()
# main.mainloop()

port = 5555
host = '127.0.0.1'

try:
    server.bind((host, port))
except socket.error as e:
    str(e)
print("socket binded to %s" %(port))

server.listen(2)
print("socket is listening, server started")


def threaded_client(conn):
    conn.send(str.encode("Connected"))
    reply = ""

    while True:
        try:
            data = conn.recv(2048)
            reply = data.decode("utf-8")

            if not data:
                print("disconnected")
                break
            else:
                print("received:", reply)
                print("sending:", reply)
            
            conn.sendall(str.encode(reply))
        except:
            break

    print("Lost connection")
    conn.close()

while True:
    conn, addr = server.accept()
    print("connection to", addr)

    start_new_thread(threaded_client, (conn,))
# while True:
#     client, addr = server.accept()
#     print("connection to", addr)

#     message = client.recv(1024)# "Thx for connecting." #
#     client.send(message.encode("utf-8"))

#     print("Closing down the server.")
#     client.close()
#     break