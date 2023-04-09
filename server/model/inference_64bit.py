import torch
import sys
import numpy as np
import chess

#sys.path.insert(1,"src/model_src_v2")
from cnn_v13_64bit_source_clear import PlainChessNET as source_net
from cnn_v13_64bit_target_clear import PlainChessNET as target_net

from helper_chess_v7_64target import Board_State, Move_State

# recall square table location within chessboard (8x8) 
    # 8 [56,57,58,59,60,61,62,63],
    # 7 [48,49,50,51,52,53,54,55],
    # 6 [40,41,42,43,44,45,46,47],
    # 5 [32,33,34,35,36,37,38,39],
    # 4 [24,25,26,27,28,29,30,31],
    # 3 [16,17,18,19,20,21,22,23],
    # 2 [8 ,9 ,10,11,12,13,14,15],
    # 1 [0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ]
    #    a  b  c  d  e  f  g  h


class Inference:


    def __init__(self):
        self.board_to_tensor = Board_State()
        self.move_to_tensor = Move_State()
        
        # instantiate the model
        self.source_model = source_net() #PlainChessNET()
        self.target_model = target_net() #PlainChessNET()


    # inference function
    def predict(self, input_board, topf=2, topt=3):
        
        dictofproposal = {}
        legal_proposal_moves = []
        source_model = self.source_model
        target_model = self.target_model

        # defining the processor
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # loading the checkpoint
        source_state_dict = torch.load("server/model/source_clear.pt",map_location = device)
        target_state_dict = torch.load("server/model/target_clear.pt",map_location = device)

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

            # convert to tensor
            chessboard, source_square_bit = torch.tensor(board).unsqueeze(0).to(torch.float).to(device), torch.tensor(source_square_bit).unsqueeze(0).to(torch.float).to(device)
            target_output = target_model(chessboard, source_square_bit)
            #target_square = torch.argmax(target_output)

            # topt target square
            _, target_square = torch.topk(target_output, topt)

            # proposal moves without legal move filter
            for t in range(topt):
                #proposal_moves.append(self.square_to_alpha(source_square.data[0][s].item(), target_square.data[0][t].item()))
                keys_prop,  values_digit = self.square_to_alpha(source_square.data[0][s].item(), target_square.data[0][t].item())

                #feeding the dict of proposal with uci format as keys and digit equivalent as values
                dictofproposal[keys_prop] = values_digit
        
        
        # from raw proposal to legal propose
        for prop, values in dictofproposal.items():

            if chess.Move.from_uci(prop) in input_board.legal_moves:
                legal_proposal_moves.append(values)
        
        print(legal_proposal_moves[0])
        return legal_proposal_moves


    def square_to_alpha(self, src_sq, trgt_sq):
        """
        convert square number into chessboard digit coordinates (due to chess lib) and alpha.
        input: source square number, target square number
        return : uci format (str), source and target as digit coordinates
        """

        # digit conversion
        col_s, row_s = chess.square_file(src_sq),chess.square_rank(src_sq)
        col_t, row_t = chess.square_file(trgt_sq),chess.square_rank(trgt_sq)

        # alpha conversion
        alpha_col_s = chess.FILE_NAMES[col_s]
        alpha_col_t = chess.FILE_NAMES[col_t]
        
        # converting coordinates to str and join to get uci move format (see chess lib)
        move_proposal = "".join((str(alpha_col_s),str(row_s+1),str(alpha_col_t),str(row_t+1)))

        return  move_proposal, ((col_s,row_s),(col_t,row_t))

