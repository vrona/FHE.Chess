import pygame
from base import *

class Game:

    def __init__(self):
        pass

    

    def show_backg(self, surface):
        for row in range(rows):
            for col in range(cols):
                if(row + col) %2 == 0:
                    color = (251,203,4) #(255,203,0,255)
                else:
                    color = (29,24,9) #(0,0,0,255)

                rect = (col * sqsize, row * sqsize, sqsize, sqsize)

                pygame.draw.rect(surface, color, rect)