import pygame
from base import *



class Dragger:

    def __init__(self):
        self.piece = None
        self.dragging = False
        self.mouseX = 0
        self. mouseY = 0
        self.initial_row = 0
        self.initial_col = 0

    def update_blit(self, surface):
        self.piece.set_texture(size=128)
        img_path = self.piece.img_uri

        # pygame specification
        img = pygame.image.load(img_path)
        img_centered = (self.mouseX, self.mouseY)
        self.piece.rectangle = img.get_rect(center=img_centered)
        surface.blit(img, self.piece.rectangle)                  


    def update_mouse(self, position):
        self.mouseX, self.mouseY = position

    def save_initial(self, position):
        self.initial_row = position[1] // sqsize # y coordinate
        self.initial_col = position[0] // sqsize # x coordinate

    def drag_piece(self, piece):
        self.piece = piece
        self.dragging = True
    
    def undrag_piece(self):
        self.piece = None
        self.dragging = False