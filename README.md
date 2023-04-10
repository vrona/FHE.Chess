
# FHE.Chess

#### WARNING
-   FHE models are not effective yet
-   AI is white only and based on clear player moves
-   Several bugs remains while playing (occured by dual chess engine: one scratch, one python-chess)

#### SET UP AND PLAY
1.   Download client & server folders into local folder <your_local_folder>. Then, $ cd <your_local_folder>
2.   Create venv based on requirements.txt and activate venv
3.   Open 2 terminals:
-       terminal 1: $ python3 server/server.py
-       terminal 2: $ python3 client/chess_env/main.py

When the Chess app crashes, execute $ python3 client/chess_env/main.py from terminal 2.


## Overview
Create an application that plays Chess against an AI oponent. The moves should be encrypted with FHE so that the AI doesn't see them but can still run its algorithm on them.

## Description
Create a machine-learning-based version of a Chess player which can be executed in FHE, i.e., where the computer does not see the unencrypted moves.
On the player (client) side, the board would be in clear; then, when she plays her move, she encrypts the new position and sends it to the server, which then runs the machine-learning model inference over encrypted data, to predict a new (encrypted) move to apply. Finally, the player decrypts this move and apply it on the position, and reiterate the process until the game is over.

### Project flow
https://github.com/vrona/FHE.Chess/blob/main/.swm/fhechess-project-flow.xf41t.sw.md


![alt text](https://github.com/vrona/FHE.Chess/blob/main/screen_zama_vrona_chess.png)