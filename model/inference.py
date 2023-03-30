import torch
import sys

sys.path.insert(1,"/Volumes/vrona_SSD/FHE.Chess/src/model_src_v2")
from cnn_v11_8bit import PlainChessNET
from helper_chess_v7_8source import Board_State


class Inference:

    def __init__(self):
        self.board_to_tensor = Board_State()
        
        # instantiate the model
        self.model = PlainChessNET()
    
    # inference function
    def predict(self, input_board, topk=3):
        model = self.model
            
        # defining the processor
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # loading the checkpoint
        state_dict = torch.load("/Volumes/vrona_SSD/FHE.Chess/weights/model_plain_chess_lj9mpw25.pt",map_location = device)

        # loading the model

        
        model.load_state_dict(state_dict)
        model.eval()

        board = self.board_to_tensor.board_tensor_12(input_board)
        
        # Prediction of the class from an image file

        ####board.requires_grad_(False)
        print(torch.tensor(board).unsqueeze(0).shape)
        
        #the_model.to(device)
        output = model(torch.tensor(board).unsqueeze(0).to(torch.float).to(device))
        ps = torch.exp(output)
        ps = ps / ps.sum()
        ps = ps ** 3
        ps = ps / ps.sum()
        """
        d2  [1.0000, 1.0000, 1.0000, 1.0000, 1.9219, 1.6073, 1.4611, 1.4126],
        00001011
        d3  [1.0000, 1.0000, 1.0000, 2.3769, 2.0349, 1.7184, 1.6319, 1.5665]
        00010011

        [0.0169, 0.0169, 0.0169, 0.0169, 0.1197, 0.0700, 0.0526, 0.0475],
        00001000
         [0.0169, 0.0169, 0.0169, 0.2264, 0.1421, 0.0856, 0.0733, 0.0648]
         00011000
         """
        
        # {0 : '0'     , 1 : '1'     , 2 : '10'    , 3 : '11'    , 4 : '100'   , 5 : '101'   , 6 : '110'   , 7 : '111',
        #  8 : '1000'  , 9 : '1001'  , 10: '1010'  , 11: '1011'  , 12: '1100'  , 13: '1101'  , 14: '1110'  , 15: '1111',
        #  16: '10000' , 17: '10001' , 18: '10010' , 19: '10011' , 20: '10100' , 21: '10101' , 22: '10110' , 23: '10111',
        #  24: '11000' , 25: '11001' , 26: '11010' , 27: '11011' , 28: '11100' , 29: '11101' , 30: '11110' , 31: '11111',
        #  32: '100000', 33: '100001', 34: '100010', 35: '100011', 36: '100100', 37: '100101', 38: '100110', 39: '100111',
        #  40: '101000', 41: '101001', 42: '101010', 43: '101011', 44: '101100', 45: '101101', 46: '101110', 47: '101111',
        #  48: '110000', 49: '110001', 50: '110010', 51: '110011', 52: '110100', 53: '110101', 54: '110110', 55: '110111',
        #  56: '111000', 57: '111001', 58: '111010', 59: '111011', 60: '111100', 61: '111101', 62: '111110', 63: '111111'}


        # [56,57,58,59,60,61,62,63],
        # [48,49,50,51,52,53,54,55],
        # [40,41,42,43,44,45,46,47],
        # [32,33,34,35,36,37,38,39],
        # [24,25,26,27,28,29,30,31],
        # [16,17,18,19,20,21,22,23],
        # [8 ,9 ,10,11,12,13,14,15],
        # [0 ,1 ,2 ,3 ,4 ,5 ,6 ,7]
        #probs, indices = ps.topk(topk)

        # indices = indices.cpu().numpy()[0]
        # idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        # classes = [idx_to_class[i] for i in indices]
        # names = [cat_to_name[str(j)] for j in classes]
        print(ps)
        return output




