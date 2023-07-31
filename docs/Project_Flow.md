# Project Flow

## Overview

An application that plays Chess against an AI opponent. The moves are encrypted then thanks to FHE, the AI infers on data that she cannot see.

## Description

Create a machine-learning-based version of a Chess player which can be executed in FHE, i.e., where the computer does not see the unencrypted moves. On the player (client) side, the board would be in clear; then, when she plays her move, she encrypts the new position and sends it to the server, which then runs the machine-learning model inference over encrypted data, to predict a new (encrypted) move to apply. Finally, the player decrypts this move and apply it on the position, and reiterate the process until the game is over.

## Knowledge

*   [**Read Me**](../README.md) provides succinct info to run the FHE.Chess. (Do Not Forget to use the requirements.txt files (see. #0 Set up - dependencies installation"): 1 for local, 1 for remote server)

*   **Semantic**: while reading you will faced to specific terms, let's cleared them out.

    *   As a chessboard is made of 64 squares (8\*8), **Source** and **Target** are respectively: the selected square of the piece to move from, the selected square of the piece to move to.

    *   **Clear**: in cryptography context, means non-encrypted.

    *   **Quantization**: refers to techniques that helps to contraint an input from continuous (floating point precision) or large set of values to a discrete set (such as integers). Two main libraries are known: _Brevitas_ and the well-known _PyTorch_.

    *   **Compilation**: is handled by Zama's Concrete-ML library. It produces low-code which acts at each computation steps within the quantized models to execute dedicated computations on encrypted data. The price of these additional operations is a slowdown at inference step. (see, "simfhe" vs "deepfhe" below). Thus, the more complex is a quantized model the longer it takes to output a prediction.

    *   **FHE circuit**: stands for Full Homomorphic Encryption which enable to compute directly on encrypted input data to infer encrypted output data.

    *   **Concrete-ML**: [Concrete ML](https://docs.zama.ai/concrete-ml/) is an open source, privacy-preserving, machine learning inference framework based on Fully        Homomorphic Encryption (FHE).

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
<div align="center"><img src="../FHE_Chess_archi_current.png" style="width:'50%'"/></div>

<br/>

*   **with client FHE on local - with server FHE on remote**: (future architecture), here the chess app itself is still in client\_local accompanied with client FHE for inputs data encryption. Then, computations on encrypted input data and inference of encrypted output data are made in remote server (instance).
<br/>
<div align="center"><img src="../FHE_Chess_archi_next.png" style="width:'50%'"/></div>

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

## #1 Chess App.

The AI needs an environment to take input from and to propose output to.
The development of the chess app itself can be done completely from scratch or based on [python-chess](https://python-chess.readthedocs.io/en/latest/) library.
It happens that this project is based on both (to speed up development).

Except the [Clone_Chess class](https://github.com/vrona/FHE.Chess/blob/quant_fhe/client_local/chess_env/clone_chess.py) which returns [python-chess](https://python-chess.readthedocs.io/en/) methods, everything from [client_local/chess_env](https://github.com/vrona/FHE.Chess/tree/quant_fhe/client_local/chess_env) is made from scratch.

## #2 Data

Data used is downloadable here: [kaggle.com/datasets/arevel](https://www.kaggle.com/datasets/arevel/chess-games)

*   **Raw data explanation** [Data Explanation](data_explanation.md)

*   ***Data preparation** is explained in this [wb_2000.ipynb](https://github.com/vrona/FHE.Chess/blob/quant_fhe/server_cloud/data/wb_2000.ipynb) notebook. _Little take away: the goal is to create an AI that would be rated at least 1500 ELO on Lichess. Thus, data preparation aimed to provide only data points from games made by chess players rated at least 2000 ELO.

*   **Data transformation** needed to nurture source and target models. Transformaions are supplied by [helper_chessset.py](https://github.com/vrona/FHE.Chess/blob/quant_fhe/server_cloud/model_src/helper_chessset.py) - detailed here [Data transformation](data_transformation.md)

## #3 Models

**The chosen philosophy is straightforward**: train one model to determine the SOURCE square (no matter the piece and evaluation), train another model to determine the TARGET square.

Consequences: AI always starts her moves the same way, but over 5 moves it starts to be very funny.

*   **#3.1 clear source / target**

    *   **Source model**

        *   input source : (12,8,8) board -> output source : selected Square number to move FROM as 1D array of shape (64,)

        *   3 convolution layers (hidden size=128) + fully-connected layer (64)

    *   **Target model**

        *   input target : (12,8,8) board + selected Square number to move from as 1D array of shape (64,) -> output target : selected Square number to move TO as 1D array of shape (64,)

        *   3 convolution layers (hidden size=128) + fully-connected layer (64)

*   **#3.2 quantized source / target**

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