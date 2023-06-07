import socket
import pickle
import sys

sys.path.insert(1,"server/model")
from model.inference_64bit import Inference

HOST = ""  # Standard loopback interface address (localhost)
PORT = 3389  # Port to listen on (non-privileged ports are > 1023)

inference = Inference()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    print("socket binded to %s" %(PORT))

    s.listen(5)
    print("socket is listening, server started")

    while True:
        conn, addr = s.accept()
        print(f"Connected by {addr}")
        while True:
            try:
                data = pickle.loads(conn.recv(2048*3))

                reply = inference.predict(data)

                if not data:
                    print("failed at data --> disconnected")
                    break
                elif not reply:
                    print("failed at reply --> disconnted")
                else:
                    print("data_from_client", data)
                    print("inference", reply)

                conn.sendall(pickle.dumps(reply))
            
            except:
                break
        
        print("Lost connection")
        conn.close()