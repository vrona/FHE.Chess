
# FHE.Chess

## Overview
Create an application that plays Chess against an AI oponent. The moves should be encrypted with FHE so that the AI doesn't see them but can still run its algorithm on them.

## Description
Create a machine-learning-based version of a Chess player which can be executed in FHE, i.e., where the computer does not see the unencrypted moves.
On the player (client) side, the board would be in clear; then, when she plays her move, she encrypts the new position and sends it to the server, which then runs the machine-learning model inference over encrypted data, to predict a new (encrypted) move to apply. Finally, the player decrypts this move and apply it on the position, and reiterate the process until the game is over.

![alt text](https://github.com/vrona/FHE.Chess/blob/main/screen_zama_vrona_chess.png)