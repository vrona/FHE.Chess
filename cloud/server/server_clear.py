import socket
import struct
import pickle

from infer_clear import Inference_clear

HOST = ""  # Standard loopback interface address (localhost)
PORT = 3389  # Port to listen on (non-privileged ports are > 1023)

inference = Inference_clear()

def recvall(conn, size):
    """letting all bytes to be received as small parts of bytes."""
    buffer = bytes()
    
    remain = size
    while remain  > 0:
        
        data_received  = conn.recv(remain)
        
        if not data_received:
            raise Exception('unexpected EOF')
        buffer += data_received
        remain -= len(data_received)

    return buffer


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
            
                data_size = struct.unpack('I', conn.recv(4))[0] 
                byte_data = recvall(conn, data_size)

            
                #print("Server side len data:",len(byte_data))

                data = pickle.loads(byte_data)
                #print("chessboard\n",data)
                reply = inference.predict(data, 5, 3)
                #print("inference\n",reply)

                if not data:
                    print("failed at data --> disconnected")
                    break
                elif not reply:
                    print("input chessboard before failed at reply\n",data)
                    #print("failed at reply --> disconnected")

                else:
                    print("input_data as chessboard")
                    print(data)
                    print("inference list of moves")
                    print(reply)

                    conn.sendall(pickle.dumps(reply))

            except :#socket.error as e:
                #print(e)
                
                break
        
        print("Lost connection")
        conn.close()


