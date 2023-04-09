import pygame
import sys


# sys.path.insert(1,"client/chess_env")
from base import sp_width, sp_height, sqsize
from game import Game
from square import Square
from move import Move
from clone_chess import Clone_Chess
from button import Button

sys.path.insert(1,"server/model")
from inference_64bit import Inference


