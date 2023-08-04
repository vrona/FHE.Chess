# Data transformation

At this step, [helper_chessset.py](../server_cloud/model_src/helper_chessset.py) supports the actions of transformation.

## Goal

This document would help the [models](model_lifecycle.md) to receive the desired input_data in correct format and transformed the ground truth data and output. This includes in training and production contexts.<br>

**Input data**<br>
The 1st layers of the models are made of Convolution Neural Network which need input_data of shape (12,8,8) and filled of binary data.<br>


* format :
    - dim 0 = a layer for each color (2) and type of pieces (6),
    - dim 1 = number of cols,
    - dim 2 = number of rows.
* binary :
    - 0: empty square
    - 1: presence of white piece
    - -1: presence of black piece

<br>

**Output**<br>

The last layers are then full connected networks layers which deliver output of shape (64,).
This document return the ground truth training data as it takes advantages of Python-Chess lib's bitboard logic.

<br>


## Recall

Ready to use "dataset": [wb_2000_300.csv](../server_cloud/data/wb_2000_300.csv) have been made due to data preparation (see. ["wb_2000" jupyter notebook](../server_cloud/data/wb_2000.ipynb)).

## The Flow involved 2 classes:
<br>

- ### Input Data (training and production)
  ```python
  class Board_State()
  ```

1.  Each move of each game (sequences of white vs black moves) from "dataset": [wb_2000_300.csv](../server_cloud/data/wb_2000_300.csv):
    ```text
    1. e4 e5 2. b3 Nf6 3. Bb2 Nc6 4. Nf3 d6 5. d3 g6 6. Nbd2 Bg7 7. g3 Be6 8. Bg2 Qd7 9. O-O O-O 10. c3 b5 11. d4 exd4 12. cxd4 Bg4 13. Rc1 Rfe8 14. Qc2 Nb4 15. Qxc7 Qxc7 16. Rxc7 Nxa2 17. Ra1 Nb4 18. Raxa7 Rxa7 19. Rxa7 Nxe4 20. Nxe4 Rxe4 21. Ng5 Re1+ 22. Bf1 Be2 23. Rxf7 Bxf1 24. Kh1 Bh3# 0-1
    ```

    is translated into chessboard like format and can be visualized thanks to [Python-Chess library](https://python-chess.readthedocs.io/en/) library: 

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

    which is then being transformed into an array of shape (12,8,8) thanks to theses two methods:
    ```python
    def feat_map_piece_12(board, color)
    def board_tensor_12(board)
    ```
    <br>

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


## Chessset dataset (PyTorch)

### dataset_source


### dataset_target