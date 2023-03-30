import torch
from cnn_v11_8bit import PlainChessNET
from helper_chess_v7_8source import Board_State



class Inference:

    def __init__(self):
        pass
    # inference function
    def predict(board, topk=3):
            
        # defining the processor
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # loading the checkpoint
        state_dict = torch.load("/Volumes/vrona_SSD/FHE.Chess/weights/model_plain_chess_lj9mpw25.pt",map_location = device)

        # instantiate the model
        model = PlainChessNET()

        # loading the model


        board_to_tensor = Board_State()
        
        model.load_state_dict(state_dict)
        model.eval()

        #board = board_to_tensor.board_tensor_12(input_board)
            
        # Prediction of the class from an image file
        
        ####board.requires_grad_(False)

        
        #the_model.to(device)
        output = model(board.to(torch.float).to(device))
        ps = torch.exp(output)

        #probs, indices = ps.topk(topk)

        # indices = indices.cpu().numpy()[0]
        # idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        # classes = [idx_to_class[i] for i in indices]
        # names = [cat_to_name[str(j)] for j in classes]
        print(ps)
        return output




