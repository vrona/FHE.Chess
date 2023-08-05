import numpy as np
from tqdm import tqdm


"""
DATA COMPLIANCE FOR CONCRETE-ML
prepare train_input data for compilation with concrete-ml
"""

"""
reminder
model 1 (source): from input data (chessboard) predict selected square to be played (source)
model 1 (source): from input data (tuple (chessboard source)) predict selected square to be played (target)
"""

def get_train_input(trainload_set, target=False):
    
    """If target=True, Concrete-ml compiles model 2 (target), otherwise model 1 (source).
    
    Goal: returning train_input as:
        - array of tensor (mono input_data)
        - tuple of arrays of tensor (multiple input_datas).
    """

    list_train_inputs = []
    list_train_sources = []
    
    loop_trainset = tqdm(enumerate(trainload_set), total=len(trainload_set), leave=False)

    if target:
        # TARGET CASE
        # preparation training input_data: chessboard, source

        for idx, (chessboard, sources, targets) in loop_trainset:
            data, source, target = chessboard.clone().detach().float(),  sources.clone().detach().float(), targets.clone().detach().float() #torch.tensor(chessboard).float(), torch.tensor(targets).float() # 

            list_train_inputs.append(data)
            list_train_sources.append(source)

        loop_trainset.set_description(f"data [{idx}/{trainload_set}]")

        train_chess = np.concatenate(list_train_inputs)
        train_source = np.concatenate(list_train_sources)

        train_input = (train_chess, train_source)
        return train_input
    
    else:
        # SOURCE CASE
        # preparation training input_data: chessboard

        for idx, (chessboard, targets) in loop_trainset:

            data, target = chessboard.clone().detach().float(), targets.clone().detach().float()
            list_train_sources.append(data)

        loop_trainset.set_description(f"data [{idx}/{trainload_set}]")

        train_input = np.concatenate(list_train_sources, axis=0)
        return train_input
