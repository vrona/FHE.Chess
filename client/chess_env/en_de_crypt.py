import sys
import chess
import torch
from concrete.ml.deployment.fhe_client_server import FHEModelClient

sys.path.insert(1,"client/")
from helper_chess_target import Board_State, Move_State

sys.path.insert(1,"server/")
from network import OnDiskNetwork

class EnDe_crypt:
    def __init__(self):
        self.source_client = "client/source"
        self.target_client = "client/target"

        self.fhesource_client = FHEModelClient(path_dir=self.source_client, key_dir=self.source_client)
        self.fhesource_client.load()

        self.fhetarget_client = FHEModelClient(path_dir=self.target_client , key_dir=self.target_client)
        self.fhetarget_client.load()

        self.board_to_tensor = Board_State()
        self.move_to_tensor = Move_State()

        self.net_fhe_work = OnDiskNetwork()


        # self.source_server = "server/source"
        # self.target_server = "server/target"

        # self.fhesource_server = FHEModelServer(path_dir=self.source_server, key_dir=self.source_server)
        # self.fhesource_server.load()

        # self.fhetarget_server = FHEModelServer(path_dir=self.target_server , key_dir=self.target_server)
        # self.fhetarget_server.load()

    """
    input: float --> quantization --> encryption
    """
    def encrypt_app(self, clear_chessboard, clear_source=None, target=False):
        
        if target:
            
            # quantized and encryption of clear target model inputs
            encrypted_input = self.fhetarget_client.quantize_encrypt_serialize(clear_chessboard, clear_source)

            return encrypted_input
        else:

            # # Quantize the values
            # quantized_x = self.fhesource_client.model.quantize_input(clear_chessboard)

            # # Encrypt the values
            # model_source = self.fhesource_client.load()
            # print(type(model_source))
            # enc_qx = model_source.encrypt(quantized_x)

            # # Serialize the encrypted values to be sent to the server
            # serialized_enc_qx = model_source.specs.serialize_public_args(enc_qx)
            # return serialized_enc_qx
            # quantized and encryption of clear source model input
            encrypted_input = self.fhesource_client.quantize_encrypt_serialize(clear_chessboard)
            return encrypted_input
    
    """
    output: float <-- dequantization <-- decryption
    """
    def decrypt(self, encrypted_output, target=False):

        if target:
            decrypted_data = self.fhetarget_client.deserialize_decrypt_dequantize(encrypted_output)
            return decrypted_data
        else:
            decrypted_data = self.fhesource_client.deserialize_decrypt_dequantize(encrypted_output)
            return decrypted_data


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
        
    # def input_to_board(self, input_board):
    #     # convert chessboard to tensor (12,8,8)
    #     board = self.board_to_tensor.board_tensor_12(input_board)

    #     return board
    
    # inference function
    def predict(self, input_board, topf=2, topt=3):
        """
        Recall: 2 disctinct models (source & target)
        input source : 12,8,8 board -> output source : selected Square number to move FROM as 1D array of shape (64,)
        input target : 12,8,8 board + selected Square number to move from as 1D array of shape (64,) -> output target : selected Square number to move TO as 1D array of shape (64,)
        """
        dictofproposal = {}
        legal_proposal_moves = []
  
        # defining the processor
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 

        # convert chessboard to tensor (12,8,8)
        self.board = self.board_to_tensor.board_tensor_12(input_board)
        """
        Prediction of source square
        """

        # from torch tensor to numpy to encryption (which includes quantization)
        chessboard = torch.tensor(self.board).unsqueeze(0).to(torch.float).to(device)
        chessboard = chessboard.cpu().detach().numpy()

        # encryption
        serialized_encrypted_chessboard = self.encrypt_app(chessboard)
        print("FLAG 1")
        # sending encrypted source for inference 1/2
        source_output_encrypted = self.net_fhe_work.run_fhe_inference(serialized_encrypted_chessboard, "/source")
        #self.net_fhe_work.client_send_input_to_server_for_prediction(serialized_encrypted_chessboard, "/source")
        print("FLAG 1.5")
        #source_output_encrypted = self.net_fhe_work.server_send_encrypted_prediction_to_client("/source")
        print("FLAG 2")
        source_output_decrypted = self.decrypt(source_output_encrypted)

        # TO FIX 3 topf source square _, source_square = torch.topk(source_output, topf)

    # def predict_target(self, input_board, source, topf=2, topt=3):
        """
        Prediction of target square of each topf source square
        """
        # convert source square number to array (64,)
        source_square_bit = self.move_to_tensor.source_flat_bit(source_output_decrypted)
    
        # convert to tensor
        chessboard, source_square_bit = torch.tensor(self.board).unsqueeze(0).to(torch.float).to(device), torch.tensor(source_square_bit).unsqueeze(0).to(torch.float).to(device)
        chessboard, source_square_bit = chessboard.cpu().detach().numpy(), source_square_bit.cpu().detach().numpy()
        # encryption
        enc_chessboard, enc_source_square = self.encrypt_app(chessboard, source_square_bit, target=True)
        
        # sending encrypted tar for inference 2/2
        target_output_encrypted = self.net_fhe_work.run_fhe_inference((enc_chessboard, enc_source_square), "/target")
        # self.net_fhe_work.client_send_input_to_server_for_prediction((enc_chessboard, enc_source_square), "/target")
        # target_output_encrypted = self.net_fhe_work.server_send_encrypted_prediction_to_client("/target")
        target_output_decrypted = self.decrypt(target_output_encrypted)

        print(source_output_decrypted, target_output_decrypted)


        # # 3 topf source square
        # _, source_square = torch.topk(source_output, topf)

        # for s in range(topf):

        # # Prediction of target square of each topf source square

        #     # convert source square number into 64 array
        #     source_square_bit = self.move_to_tensor.source_flat_bit(source_square.data[0][s].item())

        #     # convert to tensor
        #     chessboard, source_square_bit = torch.tensor(board).unsqueeze(0).to(torch.float).to(device), torch.tensor(source_square_bit).unsqueeze(0).to(torch.float).to(device)
        #     target_output = target_model(chessboard, source_square_bit)
        #     #target_square = torch.argmax(target_output)

        #     # topt target square
        #     _, target_square = torch.topk(target_output, topt)

        #     # proposal moves without legal move filter
        #     for t in range(topt):
        #         #proposal_moves.append(self.square_to_alpha(source_square.data[0][s].item(), target_square.data[0][t].item()))
        #         keys_prop,  values_digit = self.square_to_alpha(source_square.data[0][s].item(), target_square.data[0][t].item())

        #         #feeding the dict of proposal with uci format as keys and digit equivalent as values
        #         dictofproposal[keys_prop] = values_digit
        
        
        # # from raw proposal to legal propose
        # for prop, values in dictofproposal.items():

        #     if chess.Move.from_uci(prop) in input_board.legal_moves:
        #         legal_proposal_moves.append(values)
        
        # #print("Legal_Proposal",legal_proposal_moves,"\n")
        # return legal_proposal_moves