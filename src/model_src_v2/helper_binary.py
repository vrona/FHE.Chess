import numpy as np
import torch
import chess



"""
##### make a square location / binary converter as a dictionnary. 
dict_sq_binary = {}

for z in range(0, 64):
    dict_sq_binary[z] = bin(z)[2:]


# {0 : '0'     , 1 : '1'     , 2 : '10'    , 3 : '11'    , 4 : '100'   , 5 : '101'   , 6 : '110'   , 7 : '111',
#  8 : '1000'  , 9 : '1001'  , 10: '1010'  , 11: '1011'  , 12: '1100'  , 13: '1101'  , 14: '1110'  , 15: '1111',
#  16: '10000' , 17: '10001' , 18: '10010' , 19: '10011' , 20: '10100' , 21: '10101' , 22: '10110' , 23: '10111',
#  24: '11000' , 25: '11001' , 26: '11010' , 27: '11011' , 28: '11100' , 29: '11101' , 30: '11110' , 31: '11111',
#  32: '100000', 33: '100001', 34: '100010', 35: '100011', 36: '100100', 37: '100101', 38: '100110', 39: '100111',
#  40: '101000', 41: '101001', 42: '101010', 43: '101011', 44: '101100', 45: '101101', 46: '101110', 47: '101111',
#  48: '110000', 49: '110001', 50: '110010', 51: '110011', 52: '110100', 53: '110101', 54: '110110', 55: '110111',
#  56: '111000', 57: '111001', 58: '111010', 59: '111011', 60: '111100', 61: '111101', 62: '111110', 63: '111111'}

##### make a piece / binary converter as a dictionnary. 
dict_piece_binary = {
'p':'01110000', 'r':'01110010', 'n':'01101110', 'b':'01100010', 'q':'01110001', 'k':'01101011',
'P':'01010000', 'R':'01010010', 'N':'01001110', 'B':'01000010', 'Q':'01010001', 'K':'01001011'
}

# for key in dict_piece_binary.keys():
#     print(type(key))

# make XOR between square location binary vs piece binary
y=int(dict_sq_binary[63],2) ^ int(dict_piece_binary['K'],2)
y = '{0:b}'.format(y)

# convert a binary to array
new_arr = np.zeros(shape=(8,))
for i, z in enumerate(y):
    new_arr[i] = int(z)

# convert an array to tensor
y_hat = torch.tensor(new_arr, dtype=torch.int8)


# make a binary to 8 bits binary
# print(bin(y)[2:].zfill(len(dict_piece_binary['K']))) 8 bits
"""