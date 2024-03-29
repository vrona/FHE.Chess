# Project Flow

## Overview

FHE.Chess is about an application that let you play Chess against an AI opponent. The moves are encrypted and then thanks to FHE, the AI infers on data that it cannot see.

## Description

Creation of a machine-learning-based version of a Chess player which can be executed in FHE, i.e., where the AI does not see the unencrypted moves.<br>
On the player (client) side, the board would be in clear; then, when it's the AI's turn, the app encrypts the board and sends it to the server, which then runs the machine-learning model inference over encrypted data, to predict a new (encrypted) move to apply.<br>
Finally, the app decrypts this move and applies it on the board, and reiterate the process until the game is over.

## Knowledge

*   [**Read Me**](../README.md) provides succinct information to run the FHE.Chess.

*   **Semantic**: while reading, you will faced to specific terms, let's explain them out.

    *   **Bitboard**: as a chessboard is made of 64 squares (8*8), one feature of a bitboard used here is to indicate the implicit square denomination and localization as the indices from an array of shape (64,). Concretely, square "0" is located at "a1" and square "63" is at "h8". Then, to describe, for eg.: a pawn's move "a2a4", it would be from source square: 8 to target square: 24. Other deeper uses are made from [Bitboards](https://www.chessprogramming.org/Bitboards).
    
    *   **Source**, **Target**: are respectively the selected square to move from and the selected square to move to.

    *   **Clear**: means non-encrypted in cryptography context.

    *   **[Quantization](https://docs.zama.ai/concrete-ml/advanced-topics/quantization)**: refers to techniques that helps to constrain an input from continuous (floating point precision) or large set of values to a discrete set (such as integers). Two main libraries are known - _Brevitas_ and _PyTorch_ - to quantize models.

    *   **[Compilation](https://docs.zama.ai/concrete-ml/advanced-topics/compilation#fhe-simulation)**: is handled by Zama's Concrete-ML library. It produces low-code which acts at each computation steps within the quantized models to execute dedicated computations on encrypted data. The price of these additional operations is a slowdown at inference step (see, "simfhe" vs "deepfhe" below) but the huge benefit is to provide equivalent accuracy rate to non-encrypted environment. The more complex is a quantized model the longer it takes to output a prediction.

    *   **FHE circuit**: stands for Full Homomorphic Encryption which enables to compute directly on encrypted input_data to infer encrypted output data.

    *   **[Concrete ML](https://docs.zama.ai/concrete-ml/)** is an open source, privacy-preserving, machine learning inference framework based on Fully        Homomorphic Encryption (FHE).

*   **3 modes enabled** in the FHE.Chess app.:

    *   "**clear**" - the AI uses non-encrypted inputs data (current chessboard and source) and infers non-encrypted output data (the move) due to models (non-quantized).

    *   "**simfhe**" - the AI uses a simulation context `fhe="simulate"` to infer encrypted output data (the move when decrypted) based on encrypted inputs data (current chessboard and source) and thanks to quantized and compiled models.

    *   "**deepfhe**" - the AI uses the quintessence of FHE to infer encrypted output data (the move when decrypted) based on encrypted inputs data (current chessboard and source square) and thanks to quantized and compiled models.

    *   "simfhe" vs "deepfhe"

        *   the latter needs to save and deployed the models into dedicated client-server architecture. Which includes generated keys to encrypt data (client's job) and keys_evaluation to infer on encrypted data (server's job). "simfhe" simulates the said process.

        *   based on current models complexities and hardware capacity (Ice Lake CPU), unlike "simfhe" which provides an answer within a second (like "clear"), "deepfhe" takes hours to infer.

        *   both needs to have compiled models (already quantized).

        *   NB: if you test "deepfhe", you will want to kill the remote server as you will feel that the FHE.Chess "spins forever" as it waits the inferred move by the AI.<br>
        The inference time follows the VGG's one (18000 sec): see the benchmark made by Zama's machine learning team: [Deep NN for Encrypted Inference with TFHE](https://eprint.iacr.org/2023/257).
<br/>

## Architecture Client-Server

*   **current architecture**: because of local machine's OS constraint and complexity of model, see. "deepfhe" mode.<br>
Here both client-server FHE are on remote server. Basically, the chess app (scripts which runs the chessboard, pieces, movements rules, ...) itself is in `client_local`.<br>
Then, compilation, computation and inference on encrypted data are made in remote server (instance).
<br/>
<div align="center"><img src="../images/FHE_Chess_archi_current.png" style="width:'50%'"/></div>

<br/>

*   **future architecture**:<br>
Here, client FHE is on local and server FHE on remote server. The chess app itself is still in `client_local` accompanied with client FHE (to encrypt input_data). Then, computations on encrypted input_data and inference of encrypted output data are made in remote server (instance).
<br/>
<div align="center"><img src="../images/FHE_Chess_archi_next.png" style="width:'50%'"/></div>

<br/>

## Dependencies installation

_creation and activation of virtual environments are strongly recommended._
<br>

Current project run with ```python 3.8.2```
<br/>

on your local machine, run `pip install --no-cache-dir -r requirements.txt` inside `client_local` directory.
[/requirements](../requirements.txt)
```text
brevitas==0.8.0
chess==1.9.4
numpy==1.23.5
pandas==1.5.2
pygame==2.1.2
torch==1.13.1
tqdm==4.64.1

```
<br/>

on remote machine, you must:
- if on linux, run: ```sudo apt update```
- (install pip if not) run ```pip install -U pip wheel setuptools```
- run `pip install --no-cache-dir -r requirements.txt` inside `server_cloud` directory.[server_cloud/requirements](../server_cloud/requirements.txt)
```text
brevitas==0.8.0
chess==1.9.4
concrete-ml==1.0.3
numpy==1.23.5
pandas==2.0.3
torch==1.13.1
tqdm==4.64.1
wandb==0.13.10
```

<br/>

## Problematic
### AI
At the core of this project is the question: what structure would have the AI? <br>

Because we didn't want to reinvent the wheel (see well known chess engines: [Stockfish](https://stockfishchess.org) < [AlphaZero](https://arxiv.org/abs/1712.01815) < [LCZero (LeelaChessZero)](https://lczero.org))[^1] but saving money and time, a straight forward solution came up thanks to the [B. Oshri and N. Khandwala paper](http://vision.stanford.edu/teaching/cs231n/reports/2015/pdfs/ConvChess.pdf) and rationalization.

What are the indispensable points?
- the environment is a chessboard of 64 (8*8) squares, 6 types of pieces, handled by 2 opponents,
- each type of piece has an importance/value,
- each type of piece obeys to its own rule of movement (correlated with their importance),
- chess is about taking a several dimension of decisions. Based on a current context (localization of all the white and black pieces on the chessboard) and an assessment of multiple future contexts, player "white", for example, decides to select a piece from a "Source" location to a "Target" destination,
- each square of the chessboard has a value based on each piece type. (see [Piece Square Table](https://www.chessprogramming.org/Simplified_Evaluation_Function)),
- human applies specific technics or methods which would be looking for a "bad" bishop, play the "Spanish opening" or the "Sicilian defense", ...

As human has already integrated all these points, each move made by players with high rating ELO is a (spatial and time) optimization of a merge of all those points.<br>
The [Predicting Moves in Chess using Convolutional Neural Networks](http://vision.stanford.edu/teaching/cs231n/reports/2015/pdfs/ConvChess.pdf) of B. Oshri and N. Khandwal, let us already know that relevant patterns appear on recurrent context of attack and defense.<br>

Thus, **the approach** would be:
- The AI will be building on 2 deep learning models (see [Model Lifecycle doc](model_lifecycle.md)):
    - 1 to select the square where is located the piece we would like to move,
    - and only 1 to select the square of destination where the piece would move to,
- the inferred move would be filtered as ```legal_move``` by Python-Chess library's method, and then applied in the chess game environment (see [Chess_app](/docs/Chess_app/)).
- Like, B. Oshri and N. Khandwal, the rules of game and the evaluation function are not part of the input_data.

[^1]: the probability tree from a "Source"/"Target" couple is very large. <br>
The exploration of branches:
    - all branches tackled by [Alpha-Beta pruning](https://www.chessprogramming.org/Alpha-Beta) with a limited depth in the tree used by Stockfish,
    - some of them but until the very end of the game like Alpha-zero with [MCTS](https://web.archive.org/web/20180623055344/http://mcts.ai/about/index.html)
is what it takes to build a robust chess engine.

### FHE
Which data will be encrypted and use for computations?<br>
(see [Model Lifecycle doc](model_lifecycle.md))<br>
- Model 1:
    - input_data: layers for each piece type within a chessboard (spatial indication of piece's location),
    - output_data: the selected square of departure.
- Model 2:
    - input_data: layers for each piece type within a chessboard (spatial indication of piece's location) + Model 1's output,
    - output_data: the selected square of destination.

In terms of architecture, at deployment, it is necessary to base the application on the client-server canvas (see. [model_deploy_FHE](model_deploy_FHE.md)). <br>
- client: takes care of input_data encryption and decryption (thanks to private keys),
- server: takes care of the necessary computations to predict (thanks to public key).

<br>

## Data management

Raw data are downloadable here: [kaggle.com/datasets/arevel](https://www.kaggle.com/datasets/arevel/chess-games)

*   **Raw data**: quick explanation via [data Explanation](data_explanation.md),

*   **Data preparation**: is explained in this [wb_2000](https://github.com/vrona/FHE.Chess/blob/quant_fhe/server_cloud/data/wb_2000.ipynb) notebook.<br>
Little take away: the goal is to create an AI that would be rated at least 1500 ELO on Lichess.<br>
Thus, the preparation step aimed to provide only data points derived from games of chess players rated at least 2000 ELO each (white and black).

*   **Data transformation**: transformations are supplied by [helper_chessset.py](https://github.com/vrona/FHE.Chess/blob/quant_fhe/server_cloud/model_src/helper_chessset.py) for training and production. However, for compilation, [data_compliance.py](../server_cloud/server/data_compliance.py) is solicited.<br>
All details are here: [data transformation](data_transformation.md).

<br>

## Chess App.

The AI needs an environment to take input from and to apply its output.<br>
The development of the chess app itself can be done completely from scratch or based on [python-chess](https://python-chess.readthedocs.io/en/latest/) library.<br>
It happens that this project is based on both (to speed up development).<br>

Except the [Clone_Chess class](https://github.com/vrona/FHE.Chess/blob/quant_fhe/client_local/chess_env/clone_chess.py) which returns [python-chess](https://python-chess.readthedocs.io/en/) methods, everything from [client_local/chess_env](https://github.com/vrona/FHE.Chess/tree/quant_fhe/client_local/chess_env) is made from scratch.<br>

Explanations of chess app scripts are here: [Chess_app](Chess_app/Chess_app.md).

<br>

## Models lifecycle

The dedicated document to read is: [Models lifecycle](model_lifecycle.md).<br>

Sum-up, 2 models in 2 contexts:

*   **normal** (PyTorch)

    *   **[Source model](../server_cloud/model_src/clear/cnn_source_clear.py)**

        *   input source : (12,8,8) board -> output source : selected Square number to move from as 1D array of shape (64,)

        *   4 convolution layers (hidden size=128) + fully-connected layer (64)

    *   **[Target model](../server_cloud/model_src/clear/cnn_target_clear.py)**

        *   input_target : (12,8,8) board + Source model's output (selected Square number to move from) as 1D array of shape (64,) -> output target : selected Square number to move to as 1D array of shape (64,)

        *   4 convolution layers (hidden size=128) + 2 fully-connected layers (64)

*   **quantized** (Brevitas - PyTorch)

    *   **[Source model](../server_cloud/model_src/quantz/source_44cnn_quantz.py)**

        *   input source : (12,8,8) board -> output source : selected Square number to move from as 1D array of shape (64,)

        *   4 convolution layers (hidden size=128) + fully-connected layer (64)

    *   **[Target model](../server_cloud/model_src/quantz/target_44cnn_quantz.py)**

        *   input_target : (12,8,8) board + Source model's output (selected Square number to move from) as 1D array of shape (64,) -> output target : selected Square number to move to as 1D array of shape (64,)

        *   4 convolution layers (hidden size=128) + 2 fully-connected layers (64)

        *   **IMPORTANT at inference** target model diverges. The details are here at [Quantization in model_lifecycle](https://github.com/vrona/FHE.Chess/blob/main/docs/model_lifecycle.md#quantization) documentation.

*   **Results monitoring**

    Recall models configurations
    ```python
    #### Normal ####
    Epochs = 5
    Learning_rate = 1.0e-3
    criterion = nn.MSELoss()

    hidden_layers=2 # 4 CNN Layers
    hidden size=128

    #### Quantized ####
    Epochs = 10
    Learning_rate = 1.0e-3
    criterion = nn.MSELoss()

    # 4 CNN Layers
    hidden size=128
    n_bits = 4
    w_bits=4
    n_active = 84               # Maximum number of active neurons
    return_quant_tensor=True    # except the last Sigmoid activation
    ```

    *  **Training and Validation losses**

    Normal and quantized models' training, validation results show that models are very close and the latter needs, at one point, more time to learn.<br>
    Indeed, as there is a lost of float precision and not all the neurons are activate, models have been trained twice longer only to scrape together more precision.<br>
    It enabled to keep the slope of learning while keeping important parameters such as learning_rate, number of hidden_layers and criterion identical.
    <br>

    You will noticed that slightly difference of precision in quantization shows bigger gap between models at accuracy test step.<br>
    
    Below, the visualizations display losses differences where for:
    - Source models: Orange are Normal (aka not-quantized) models, Green are quantized ones,
    - Target models: Orange are Normal (aka not-quantized) models, Blue are quantized ones.

    *   **Source**

        <div align="center"><img src="../images/train_losses_source.png" style="width:'50%'"/></div><br>
        
        <div align="center"><img src="../images/valid_losses_source.png" style="width:'50%'"/></div><br>


    *   **Target**
        
        <div align="center"><img src="../images/train_losses_target.png" style="width:'50%'"/></div><br>
        
        <div align="center"><img src="../images/valid_losses_target.png" style="width:'50%'"/></div><br>


    *   **Model's accuracies**<br>
    
    Here, the graph curves show **accuracies of clear vs fhe (simulation) inferences on the same 81000+ moves testset**.
    - Source models: the Greens.<br>
        Model under fhe simulated context is about 2% less accurate (45% vs 46% under clear) and this is because of quantization (see above),

    - Target models: the Blues.<br>
        The gap is increased up to 5% (52.6% fhe simulated vs 55% clear).<br>
        Here, in addition of quantization (see above) the gap is bigger perhaps because how the *Brevitas lib* handles arithmetic operation on QuantTensors.<br>
        Indeed, the precision acquired while learning by two QuantTensors may be partially vanished when adding them. And this is because one of the QuantTensor's inner scale is replaced by the other (scales has to be the same).<br>
        (Some have trouble concatenating QuantTensor with Brevitas, as well).

    <div align="center"><img src="../images/accuracy_all.png" style="width:'50%'"/></div><br>

    The gap between Target model global accuracy vs Source model is substantial due to the combination of the input_data: chessboard + source square.<br>
    The pattern between this merged input (with normalized value) and the output target square (from training data) is simpler to converge to than a chessboard input_data and a source square as an output (aka Source model's job).

## Compilation / Simulation / Deployment (FHE client-server)

Testing of quantized models above involves Compilation and Simulation steps.

- Compilation and simulation are described in [compilation](compilation.md) documentation where all the needed scripts are linked. A specific work on data is explained in [data_transformation](data_transformation.md).
- :o: To run the "deepfhe" mode, you must deploy. Deployment is powered by [client_server_fhe_deploy.py](../server_cloud/client_server_fhe_deploy.py) and explained in [model_deploy_FHE](model_deploy_FHE.md) documentation.

## Set up and play

As the app is based on a client-server architecture, client is at local, server at remote instance.<br>
(not to be confused with client-server architecture used when deploying models under the context of FHE).

<br>

**Local**
<br>
1.   ```mkdir client_local``` directory on your local machine (macOS, Linux, Windows),
2.   Create venv based on the [/requirements.txt](requirements.txt) file and activate venv,
3.   Download the content of ```client_local``` into your ```client_local``` local directory,
4.   ```cd client_local```
<br>

**Remote instance**
1.   Create a remote instance that runs under Intel Ice Lake CPU. Name of instance in GCI: "n2-standard-8", in AWS: EC2 "M6i",
2.   Run the remote instance and grab: public **IP_address** + **port** that enables to communicate with instance under firewall constrains (**for eg.: GCI, port 3389**),
3.   Create an SSH connection due to another terminal to command your remote instance. (if you don't know how, see [^2])<br>
4.   Create venv based on the [server_cloud/requirements.txt](server_cloud/requirements.txt) file and activate venv,
5.   ```mkdir fhechess``` directory,
6.   Download the content of ```server_cloud``` **_(without the mentioned large files)_** into ```fhechess``` directory.
7.   ```cd fhechess```.

At this step, you have 2 different terminals which are running simultaneously.<br>
Then, run:
<br>

**1st remote terminal**: ```$ python3 server/server_all.py -i (or --inference) "clear" or "simfhe" or "deepfhe"```<br>
!! Wait until the server is connected !! (waiting time:```"clear"``` and ```"deepfhe"``` < several seconds, ```"simfhe"``` between 2 and 7 mins)<br>

**2nd local terminal**: ```$ python3 client_local/chess_env/main.py --server IP_address --port PORT```<br>
NB:
- ```--server```: **Required option** and it enables "White AI" and "Black AI" modes,
If you have mistyped your IP_Address or if you forgot to run your remote server, please answer to the prompt displayed by your Local Terminal.
- ```--port```: **Facultative** if your value is the default value:```3389```. This is the ok firewall on GCI and AWS.
<br>

There is a "developer mode" called ```--devmode```. **Facultative** if you are not interesting in:
- "Human vs Human" game,
- "AI vs AI"game.<br>
Its default value: ```False```.<br>
To activate it, run ```$ python3 client_local/chess_env/main.py --devmode True --server "IP_address" --port PORT```
<br>

## Reset and kill
- to reset the game: press r,
- to kill: ctrl+C on local terminal or close the pygame window.
- in deepfhe mode, as it takes hours to predict (see. **mode** explanation [Project Flow](docs/Project_Flow.md)), kill the remote terminal.
<br>

[^2]: if needed, main steps to create ssh connection with GCI on Linux/Mac:
    -  recall your ```USERNAME``` from GCI and think about a ```KEY_FILENAME```,
    -  create your keys: run the command ```ssh-keygen -t rsa -f ~/.ssh/KEY_FILENAME -C USERNAME -b 2048``` (to see them ```ls .ssh```),
    -  then copy the private key (.pub is the public one)
    -  Add private key + ```USERNAME``` into your instance's metadata (follow this [process](https://cloud.google.com/compute/docs/connect/add-ssh-keys?hl=fr#add_ssh_keys_to_instance_metadata)),  
    -  ```cd .ssh```,
    -  established ssh connection with your instance, run ```ssh -i KEY_FILENAME USERNAME@IP_address```.
