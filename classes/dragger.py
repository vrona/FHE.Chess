import pygame
from base import *



class Dragger:

    def __init__(self):
        
        self.mouseX = 0
        self. mouseY = 0

    def update_mouse(self, position):
        self.mouseX, self.mouseY = position


