import pygame
from base import *

# class to let pygame's mouse grabs pieces
class Dragger:

    def __init__(self):
        self.piece = None
        self.dragging = False
        self.mouseX = 0
        self.mouseY = 0
        self.source_row = 0
        self.source_col = 0

    # with grabbing piece move from 80px to 128px (UX stuff)
    def update_blit(self, surface):
        """user experience stuff"""
        self.piece.set_texture(size=128)
        img_path = self.piece.img_uri

        # pygame specification
        img = pygame.image.load(img_path)
        img_centered = (self.mouseX, self.mouseY)
        self.piece.rectangle = img.get_rect(center=img_centered)
        surface.blit(img, self.piece.rectangle)                  

    # for grabbing
    def update_mouse(self, position):
        """current mouse position helper"""
        self.mouseX, self.mouseY = position

    def save_source(self, position):
        """get source position of square"""
        self.source_row = position[1] // sqsize # y coordinate
        self.source_col = position[0] // sqsize # x coordinate

    def drag_piece(self, piece):
        self.piece = piece
        self.dragging = True
    
    def undrag_piece(self):
        self.piece = None
        self.dragging = False