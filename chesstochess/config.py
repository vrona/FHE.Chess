import pygame
import os
from sound import Sound


class Config:

    def __init__(self):
        
        #font
        self.font = pygame.font.SysFont('menlo', 14, bold=True)
        #sound causes crashes
        #self.move_sound = Sound(os.path.join('content/sounds/move.wav'))
        #self.capture_sound = Sound(os.path.join('content/sounds/capture.wav'))

