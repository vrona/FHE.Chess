import socket
from _thread import *
import sys
import pickle

sys.path.insert(1,"server/model")
from inference_64bit import Inference

sys.path.insert(1,"client/chess_env")
from clone_chess import Clone_Chess

# instantiation
clone_chess = Clone_Chess()
inference = Inference()
# socket
server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)


port = 5555
host = '127.0.0.1'

# bind
try:
    server.bind((host, port))
except socket.error as e:
    str(e)
print("socket binded to %s" %(port))

# listen
server.listen(2)
print("socket is listening, server started")

board_from_space = clone_chess.get_board()

def threaded_client(conn):
    conn.send(pickle.dumps(board_from_space))
    reply = ""

    while True:
        try:
            data = pickle.loads(conn.recv(2048))
            reply = inference.predict(data)

            if not data:
                print("disconnected")
                break
            else:
                #print("received:", data)
                print("sending:", reply)
            
            conn.sendall(pickle.dumps(reply))
        except:
            break

    print("Lost connection")
    conn.close()

# accept
while True:
    conn, addr = server.accept()
    print("connection to", addr)

    start_new_thread(threaded_client,(conn, ))
