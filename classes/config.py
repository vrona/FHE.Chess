import pygame
import os
from sound import Sound


class Config:

    def __init__(self):
        
        #font
        self.move_sound = Sound(os.path.join('content/sounds/move.wav'))
        self.capture_sound = Sound(os.path.join('content/sounds/capture.wav'))

