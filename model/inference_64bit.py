import torch
import sys
import numpy as np
import chess

sys.path.insert(1,"/Volumes/vrona_SSD/FHE.Chess/src/model_src_v2")
from cnn_v13_64bit_source_unfhe import PlainChessNET as source_net
from cnn_v13_64bit_target_unfhe import PlainChessNET as target_net

from helper_chess_v7_64target import Board_State, Move_State

# array of square table location within chessboard (8x8) 
bitboard = np.array([
    [56,57,58,59,60,61,62,63],
    [48,49,50,51,52,53,54,55],
    [40,41,42,43,44,45,46,47],
    [32,33,34,35,36,37,38,39],
    [24,25,26,27,28,29,30,31],
    [16,17,18,19,20,21,22,23],
    [8,9,10,11,12,13,14,15],
    [0,1,2,3,4,5,6,7],
    ])

algebraic_notation_cols = {0:"a", 1:"b", 2:"c",  3:"d",  4:"e",  5:"f",  6:"g",  7:"h"}

class Inference:

    def __init__(self):
        self.board_to_tensor = Board_State()
        self.move_to_tensor = Move_State()
        
        # instantiate the model
        self.source_model = source_net() #PlainChessNET()
        self.target_model = target_net() #PlainChessNET()
    
    # inference function
    def predict(self, input_board, topf=3, topt=2):

        source_model = self.source_model
        target_model = self.target_model

        # defining the processor
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # loading the checkpoint
        source_state_dict = torch.load("/Volumes/vrona_SSD/FHE.Chess/weights/source_run-20230331_070007-igpm68ny/model_plain_chess4.pt",map_location = device)
        target_state_dict = torch.load("/Volumes/vrona_SSD/FHE.Chess/weights/target_run-20230331_113001-ksiulsjk/target_model_plain_chess_4.pt",map_location = device)

        # loading models
        ## source
        source_model.load_state_dict(source_state_dict)
        source_model.eval()

        ## target
        target_model.load_state_dict(target_state_dict)
        target_model.eval()

        # chessboard
        board = self.board_to_tensor.board_tensor_12(input_board)
        
        # Prediction of source square
        #print(torch.tensor(board).unsqueeze(0).shape)
        
        #the_model.to(device)
        source_output = source_model(torch.tensor(board).unsqueeze(0).to(torch.float).to(device))
        #source_square = torch.argmax(source_output)

        # 3 topf source square
        _, source_square = torch.topk(source_output, topf)

        for s in range(topf):

        # Prediction of target square of each topf source square
            # convert source square number into 64 array
            source_square_bit = self.move_to_tensor.source_flat_bit(source_square.data[0][s].item())
            chessboard, source_square_bit = torch.tensor(board).unsqueeze(0).to(torch.float).to(device), torch.tensor(source_square_bit).unsqueeze(0).to(torch.float).to(device)
            target_output = target_model(chessboard, source_square_bit)
            #target_square = torch.argmax(target_output)

            # 2 topt target square
            _, target_square = torch.topk(target_output, topt)

            for t in range(topf):
                #print(source_square.data[0][s].item(),"-->",target_square.data[0][t].item())
                self.square_to_alpha(source_square.data[0][s].item(),target_square.data[0][t].item())

        # source_square = source_square / source_square.sum()
        # source_square = source_square ** 3
        # source_square = source_square / source_square.sum()


        # [56,57,58,59,60,61,62,63],
        # [48,49,50,51,52,53,54,55],
        # [40,41,42,43,44,45,46,47],
        # [32,33,34,35,36,37,38,39],
        # [24,25,26,27,28,29,30,31],
        # [16,17,18,19,20,21,22,23],
        # [8 ,9 ,10,11,12,13,14,15],
        # [0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ]
        #probs, indices = source_square.topf(topf)

        # indices = indices.cpu().numpy()[0]
        # idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        # classes = [idx_to_class[i] for i in indices]
        # names = [cat_to_name[str(j)] for j in classes]
        #print(source_square, target_square)
        return source_output #, target_square
    

    def square_to_alpha(self,src_sq, trgt_sq):
        print(chess.square_file(src_sq), chess.square_file(trgt_sq))
        #print(square,">>>",algebraic_notation_cols[chess.square_file(square)],chess.square_rank((square)))

    def square_to_alphadigit(self):#, source, target

        dict_chess = {}
        for r in range(8):
            for c in range(8):
                dict_chess[bitboard[7-r][c]] = (r,c)

        return dict_chess

        


#def get_algeb_not(col):

    #return algebraic_notation_cols[col]






infer = Inference()
infer.square_to_alphadigit()
