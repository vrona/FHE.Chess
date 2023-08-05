# Project Flow

## Overview

An application that plays Chess against an AI opponent. The moves are encrypted then thanks to FHE, the AI infers on data that it cannot see.

## Description

Create a machine-learning-based version of a Chess player which can be executed in FHE, i.e., where the computer does not see the unencrypted moves. On the player (client) side, the board would be in clear; then, when it plays its move, it encrypts the new position and sends it to the server, which then runs the machine-learning model inference over encrypted data, to predict a new (encrypted) move to apply. Finally, the player decrypts this move and apply it on the position, and reiterate the process until the game is over.

## Knowledge

*   [**Read Me**](../README.md) provides succinct information to run the FHE.Chess.

*   **Semantic**: while reading, you will faced to specific terms, let's clear them out.

    *   As a chessboard is made of 64 squares (8*8), **Source** and **Target** are respectively: the selected square to move from, the selected square to move to.

    *   **Clear**: means non-encrypted in cryptography context.

    *   **Quantization**: refers to techniques that helps to constrain an input from continuous (floating point precision) or large set of values to a discrete set (such as integers). Two main libraries are known - _Brevitas_ and _PyTorch_ - to quantize models.

    *   **Compilation**: is handled by Zama's Concrete-ML library. It produces low-code which acts at each computation steps within the quantized models to execute dedicated computations on encrypted data. The price of these additional operations is a slowdown at inference step (see, "simfhe" vs "deepfhe" below) but provide equivalent accuracy rate to non-encrypted environment. Thus, the more complex is a quantized model the longer it takes to output a prediction.

    *   **FHE circuit**: stands for Full Homomorphic Encryption which enable to compute directly on encrypted input data to infer encrypted output data.

    *   **[Concrete ML](https://docs.zama.ai/concrete-ml/)** is an open source, privacy-preserving, machine learning inference framework based on Fully        Homomorphic Encryption (FHE).

*   **3 modes enabled** in the FHE.Chess app.:

    *   "**clear**" - the AI uses non-encrypted inputs data (current chessboard and source) and infers non-encrypted output data (the move) due to non quantized model.

    *   "**simfhe**" - the AI uses a simulation context `fhe="simulate"` to infer decrypted output data (the move) based on encrypted inputs data (current chessboard and source) and thanks to quantized and compiled models.

    *   "**deepfhe**" - the AI uses the quintessence of FHE to infer decrypted output data (the move) based on encrypted inputs data (current chessboard and source) and thanks to quantized and compiled models.

    *   "simfhe" vs "deepfhe"

        *   the latter needs to save and deployed the models into dedicated client-server architecture. Which includes generated keys to encrypted data (client's job) and keys\_evalutation to infer on encrypted data (server's job). "simfhe" simulates the said process.

        *   based on current model complexity and hardware capacity (Ice Lake CPU), unlike "simfhe" which provides an answer within the milliseconds (like "clear"), "deepfhe" takes hours to infer.

        *   both needs to have compiled (quantized) models.

        *   NB: if you test "deepfhe", you will want to kill the remote server as the FHE.Chess will "spinning forever" as it waits the inferred move by the AI.
<br/>

## Architecture Client-Server

*   **with both client-server FHE on remote**: (current architecture due to local machine's OS constraint and complexity of model, see. "deepfhe" mode), basically the chess app (scripts which runs the chessboard, pieces, movements rules, ...) itself is in `client_local`. Then, compilation, computation and inference on encrypted data (due to Concrete-ML library) are made in remote server (instance).
<br/>
<div align="center"><img src="../images/FHE_Chess_archi_current.png" style="width:'50%'"/></div>

<br/>

*   **with client FHE on local - with server FHE on remote**: (future architecture), here the chess app itself is still in client\_local accompanied with client FHE for inputs data encryption. Then, computations on encrypted input data and inference of encrypted output data are made in remote server (instance).
<br/>
<div align="center"><img src="../images/FHE_Chess_archi_next.png" style="width:'50%'"/></div>

<br/>

<br/>

## #0 Set up - dependencies installation

_creation and activation of virtual environments are strongly recommended._
<br/>

on your local machine, run `pip install --no-cache-dir -r requirements.txt` inside `client_local` directory.
[/requirements](../requirements.txt)
```text
1      chess==1.9.4
2      numpy==1.23.5
3      pygame==2.1.2
```
<br/>

on remote machine, run `pip install --no-cache-dir -r requirements.txt` inside `server_cloud` directory.
[server_cloud/requirements](../server_cloud/requirements.txt)
```text
1      brevitas==0.8.0
2      chess==1.9.4
3      concrete-ml==1.0.3
4      numpy==1.23.5
5      pygame==2.1.2
6      torch==1.13.1
7      tqdm==4.64.1
8      wandb==0.13.10
```

<br/>

## #1 Problematic
### AI
At the core of this project is the question: what structure would have the AI? <br>

Because we didn't want to reinvent the wheel (see well known chess engines: [Stockfish](https://stockfishchess.org) < [AlphaZero](https://arxiv.org/abs/1712.01815) < [LCZero (LeelaChessZero)](https://lczero.org)) but saving money and time, a straight forward solution came up thanks to the [B. Oshri and N. Khandwala paper]((http://vision.stanford.edu/teaching/cs231n/reports/2015/pdfs/ConvChess.pdf)) and rationalization.

What are the indispensable points?
- the environment is a chessboard of 64 (8*8) squares, 6 types of pieces, handled by 2 opponents,
- each type of piece has an importance/value,
- each type of piece obeys to its own rule of movement (correlated with their importance),
- chess is about taking a several dimension of decisions. Based on a current context (localization of all the white and black pieces on the chessboard) and an assessment of multiple future contexts, White decides to selects a piece from a "Source" location to a "Target" destination.
- to tend to a specific context, the probability tree from a "Source"/"Target" couple is very large. <br>
The exploration of branches (all branches tackled by [Alpha-Beta pruning](https://www.chessprogramming.org/Alpha-Beta) with a limited depth in the tree used by Stockfish, or some of them but until the very end of the game like Alpha-zero with [MCTS](https://web.archive.org/web/20180623055344/http://mcts.ai/about/index.html)) is what it takes to build a robust chess engine, **LCZero.... TO FINISH**
- each square of the chessboard has a value based on each piece type. (see [Piece Square Table](https://www.chessprogramming.org/Simplified_Evaluation_Function)).
- human applies specific technics or methods which would be looking for a "bad" bishop, play the "Spanish opening" or the "Sicilian defense", ...

As human has already integrated all these points, each move made by player with high rating ELO is an optimization of a merge of all those points. <br>
[Predicting Moves in Chess using Convolutional Neural Networks](http://vision.stanford.edu/teaching/cs231n/reports/2015/pdfs/ConvChess.pdf), let us already know that relevant patterns appear on recurrent context of attack and defense. <br>
In addition, we learn about their method that the rules of game and the evaluation function are not part of the input_data.

Thus, **the approach** would be:
- The AI will be building on 2 deep learning models (see [Model Lifecycle doc](model_lifecycle.md)):
    - 1 to select the square where is located the piece we would like to move,
    - and only 1 to select the square of destination where the piece would move to,
- the inferred move would be filtered as ```legal_move``` by Python-Chess library's method, and then applied in the chess game environment (see [Chess_app](/docs/Chess_app/)).

### FHE
Which data will be encrypted and use for computations?<br>
(see [Model Lifecycle doc](model_lifecycle.md))<br>
- Model 1:
    - input_data: the pieces on the chessboard (spatial indication of piece's location),
    - output_data: the selected square of departure.
- Model 2:
    - input_data: the pieces on the chessboard (spatial indication of piece's location) + Model 1's output,
    - output_data: the selected square of destination.

In terms of architecture, at deployment, it is necessary to base the application on the client-server canvas. <br>
- client: takes care of input_data encryption and decryption (keys generation),
- server: takes care of the necessary computations to predict (key evaluation).

<br>

## #2 Data

Raw data are downloadable here: [kaggle.com/datasets/arevel](https://www.kaggle.com/datasets/arevel/chess-games)

*   **Raw data explanation**: see [Data Explanation](data_explanation.md)

*   **Data preparation**: is explained in this [wb_2000](https://github.com/vrona/FHE.Chess/blob/quant_fhe/server_cloud/data/wb_2000.ipynb) notebook.<br>
Little take away: the goal is to create an AI that would be rated at least 1500 ELO on Lichess.<br>
Thus, the preparation step aimed to provide only data points from games derived from chess players rated at least 2000 ELO each (white and black).

*   **Data transformation**: Transformations are supplied by [helper_chessset.py](https://github.com/vrona/FHE.Chess/blob/quant_fhe/server_cloud/model_src/helper_chessset.py) - detailed here [Data transformation](data_transformation.md)

<br>

## #2 Chess App.

The AI needs an environment to take input from and to propose output to.
The development of the chess app itself can be done completely from scratch or based on [python-chess](https://python-chess.readthedocs.io/en/latest/) library.
It happens that this project is based on both (to speed up development).

Except the [Clone_Chess class](https://github.com/vrona/FHE.Chess/blob/quant_fhe/client_local/chess_env/clone_chess.py) which returns [python-chess](https://python-chess.readthedocs.io/en/) methods, everything from [client_local/chess_env](https://github.com/vrona/FHE.Chess/tree/quant_fhe/client_local/chess_env) is made from scratch.<br>

Explanations of chess app scripts are here [Chess_app](Chess_app/Chess_app.md).

## #3 Models

Sum-up, 2 models in 2 contexts:

*   **# clear source / target**

    *   **Source model**

        *   input source : (12,8,8) board -> output source : selected Square number to move FROM as 1D array of shape (64,)

        *   4 convolution layers (hidden size=128) + fully-connected layer (64)

*   **Target model**

    *   input_target : (12,8,8) board + selected Square number to move from as 1D array of shape (64,) -> output target : selected Square number to move TO as 1D array of shape (64,)

    *   3 convolution layers (hidden size=128) + fully-connected layer (64)

*   **# quantized source / target**

    *   **Source model**

            *   input source : (12,8,8) board -> output source : selected Square number to move FROM as 1D array of shape (64,)

            *   4 convolution layers (hidden size=128) + fully-connected layer (64)

    *   **Target model**

        *   input_target : (12,8,8) board + selected Square number to move from as 1D array of shape (64,) -> output target : selected Square number to move TO as 1D array of shape (64,)

## #4 Train / Validation / Test

*   Clear models: Train/Validation

<br/>

<div align="center"><img src="https://firebasestorage.googleapis.com/v0/b/swimmio-content/o/repositories%2FZ2l0aHViJTNBJTNBRkhFLkNoZXNzJTNBJTNBdnJvbmE%3D%2F43a2be88-898c-436d-9c64-dce01319ef35.png?alt=media&token=26f73975-8ca4-41ee-96e5-06f8ea44a82b" style="width:'50%'"/></div>

<br/>

*   Clear models: Test/Accuracy

<br/>

<div align="center"><img src="https://firebasestorage.googleapis.com/v0/b/swimmio-content/o/repositories%2FZ2l0aHViJTNBJTNBRkhFLkNoZXNzJTNBJTNBdnJvbmE%3D%2Fd3f2d163-d3cb-4b7a-8a80-1269a3d5ccf6.png?alt=media&token=1dba49d2-0a8d-45a8-bf70-c5331dc1cfb3" style="width:'50%'"/></div>

<br/>

*   Quantized models: Train/Validation

*   Quantized models: Test/Accuracy

## #5 Model Quantization

## #6 Simulation

## #7 Compilation

## #8 Deployment (client-server)

## #9 Set up and play

1\. Download `client_local` on your local machine & then dowload the content of the`server_cloud` folders into local folder `<your_local_folder>`. Then, `$ cd <your_local_folder>`

2\. Create venv based on requirements.txt and activate venv

3\. Open 2 terminals:

\- terminal 1: `$ python3 server/server.py`

\- terminal 2: `$ python3 client/chess_env/main.py`

**_NB._**: When the Chess app crashes, execute `$ python3 client/chess_env/main.py` from terminal 2.

<br/>
