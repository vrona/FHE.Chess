# Data transformation

At this step, [helper_chessset.py](../server_cloud/model_src/helper_chessset.py) supports the actions of transformation.

## Goal

This script would help the [models](model_lifecycle.md) to receive the desired input_data in correct format and transformed the ground truth data and output. This concerns training and inference contexts.<br>

**Input data**<br>
The 1st layers of the models are made of Convolution Neural Network which need input_data of shape (12,8,8) and filled of binary data.<br>


* format :
    - dim 0 = a layer for each color (2) and type of pieces (6),
    - dim 1 = number of cols (8),
    - dim 2 = number of rows (8).
* binary :
    - 0: empty square
    - 1: presence of white piece
    - -1: presence of black piece

<br>

**Output**<br>
The last layers are then full connected networks layers which deliver output as an array of shape (64,).
Due to Python-Chess lib's bitboard logic, this script returns the ground truth training data.<br>


## Recall

Ready to use "dataset": [wb_2000_300.csv](../server_cloud/data/wb_2000_300.csv) have been made due to data preparation (see. ["wb_2000" jupyter notebook](../server_cloud/data/wb_2000.ipynb)).<br><br>

## The flow involved 2 classes
<br>

- ### Input Data (training and production)
  ```python
  class Board_State()
  ```

1.  Each move of each game (sequences of white vs black moves) from "dataset": [wb_2000_300.csv](../server_cloud/data/wb_2000_300.csv):
    ```text
    1. e4 e5 2. b3 Nf6 3. Bb2 Nc6 4. Nf3 d6 5. d3 g6 6. Nbd2 Bg7 7. g3 Be6 8. Bg2 Qd7 9. O-O O-O 10. c3 b5 11. d4 exd4 12. cxd4 Bg4 13. Rc1 Rfe8 14. Qc2 Nb4 15. Qxc7 Qxc7 16. Rxc7 Nxa2 17. Ra1 Nb4 18. Raxa7 Rxa7 19. Rxa7 Nxe4 20. Nxe4 Rxe4 21. Ng5 Re1+ 22. Bf1 Be2 23. Rxf7 Bxf1 24. Kh1 Bh3# 0-1
    ```

    is translated into chessboard string matrix like format and can be visualized thanks to [Python-Chess library](https://python-chess.readthedocs.io/en/) library: 

    ```
    r . b q k b . r
    p p p p . Q p p
    . . n . . n . .
    . . . . p . . .
    . . B . P . . .
    . . . . . . . .
    P P P P . P P P
    R N B . K . N R
    ```
    (This format let you visualize up to a specific move all the implicit historical moves that have been done and provides the current chessboard.)<br>

    This ```chess.Board()``` is then transformed into an tensor of shape (12,8,8) thanks to theses two methods:
    ```python
    def feat_map_piece_12(board, color)
    def board_tensor_12(board)
    ```
    **NB: methods for 6 pieces (1 layer for each piece type but both color are merged) has been written and tested. They've been left in the script to let them been used if anyone wanted to.**<br>
    
    ```python
    def feat_map_piece_6(board, color)
    def board_tensor_6(board)
    ```
    <br>


- ### Ground truth Output Data
  ```python
  class Move_State()
  ```
    - training
    ```python
    def from_to_bitboards(move, board)
    ```
    This method converts moves training elements: the source ("from" square) and the target ("to" square) into output format (array (64,)).<br>
    For eg.:
    - white moves e2e4 is 'pawn' source = e2 and target = e4.<br>
    - under the logic of bitboard (see variable ```bitboard```), this move becomes source = 12 and target = 28
    - as an array:
    ```python
    # source
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    # target
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    ```
    <br>

    Model 1 (called source) will use ```source_flat_bit``` for its output and the ground truth data.<br>
    Model 2 (called target) will use ```target_flat_bit```.<br>

    **NB**: Target model uses an additional input data which is the ```source_flat_bit```.


    - production

    Target model needs the ```source_flat_bit()``` method to convert the square number of source (following the example above, it would be 12) into an array (64,).

<br>

## Chessset dataset (PyTorch)

[PyTorch](https://pytorch.org) provides a class that helps to create easily a bridge between dataset and models.

[dataset_source.py](../server_cloud/model_src/dataset_source.py) is used for Source model.<br>
[dataset_target.py](../server_cloud/model_src/dataset_target.py) is used for Target model.<br>

Their classes are used by at:
- training, validation and testing:
  - clear: [launch_train_test_clear.py](../server_cloud/traintest_only/launch_train_test_clear.py)
  - quantz: [launch_train_quantz.py](../server_cloud/traintest_only/launch_train_quantz.py), [launch_(test)_compile_fhe.py](../server_cloud/traintest_only/launch_(test)_compile_fhe.py)
- compilation:
  - simulation fhe: [compile_fhe_inprod.py](../server_cloud/server/compile_fhe_inprod.py)
  - fhe: [client_server_fhe_deploy.py](../server_cloud/client_server_fhe_deploy.py)


2 methods are worth to talk about:
- ```python
  def __len__()
  ```
  returns the size of dataset that will be loaded by another  PyTorch's method.<br>
  The main dataset is splitted into 3 sub dataset: training (60% of the main dataset), validation and testing (20% each).<br>

- ```python
  def __getitem__(idx)
  # get the game
    random_game = self.games_df.values[idx]
    initial_moves = helper_move_state.list_move_sequence(random_game)

    # get random move
    game_state_i = np.random.randint(len(initial_moves)-1)
    next_move = initial_moves[game_state_i]  # piece_pos

    # get the sequence of moves until the move
    moves = initial_moves[:game_state_i]

    # instantiate board from chess lib
    board = chess.Board()

    for move in moves:
        board.push_san(move)

    x = helper_board_state.board_tensor_12(board)          # shape(6,8,8) or shape(12,8,8)
    
    y, _ = helper_move_state.from_to_bitboards(next_move, board) # shape (1)

    # determine white or black turn (1 for w, -1 for b) and then the one to play has always positive value
    if game_state_i %2 == 1:
        x *= -1

    return x, y
  ```
  returns the input_data (x) as a binary tensor (12,8,8) and ground_truth_data (source: y, _ or target: y, t) where y is the source square for source model and t (target square for target model).<br>

  The flow is basically: a random game > random move > get the historical moves until the selected move > push history and move into Python-Chess board to get chessboard matrix > transform matrix into binary tensor and flat binary array.


## Special step: Compilation

When it is about compilation step. Training_data must be np.array type.<br>
This is where [data_compliance.py](../server_cloud/server/data_compliance.py) comes in.<br>

It basically uses ```Dataloader``` (PyTorch's method) and then each data (Source model: input_data_1, ground_truth, Target model: input_data_1, input_data_2, ground_truth) is converted (from tensor) to np.array.<br>
A minimum of 100 training data points are necessary when compiling a model.

