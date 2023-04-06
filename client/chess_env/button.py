import pygame
from base import *

class Button:
	
    def __init__(self, y = sp_height/2 -30):

        self.normal = True
        self.y_pos = y
        self.mode = "batman"#(False, "")


    def get_mode(self):
        return self.mode
    

    def ckeck_click(self, x, action_name):

        mouse_pos = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()[0]
        button_rect = pygame.rect.Rect((x, self.y_pos),(135,30))
        
        if click and button_rect.collidepoint(mouse_pos) and self.normal:
            # set the AI or Human mode
            self.mode = action_name
            self.normal = False

        else:
           return False


    def draw(self, surface, color, text, x):

        self.text = text
        font = pygame.font.Font('freesansbold.ttf', 26)
        button_text = font.render(self.text, True, 'black')
        button_rect = pygame.rect.Rect((x, self.y_pos),(135,30))
        pygame.draw.rect(surface, color, button_rect, 0, 5)
        pygame.draw.rect(surface, 'black', button_rect, 2, 3)
        surface.blit(button_text, (x+3, self.y_pos +3))


    def button_whiteAI(self, surface):
        self.button_name = 'WHITE AI'
        self.ckeck_click(98.75, self.button_name)
        if self.normal:
            self.draw(surface,'white', self.button_name,98.75)


    # def button_blackAI(self, surface):
    #     self.button_name = 'BLACK AI'
    #  
    #     self.ckeck_click(332.5, self.button_name)
    #     if self.normal:
    #         self.draw(surface, 'white', self.button_name, 332.5)


    # def button_bothAI(self, surface):
    #     self.button_name = 'AI vs AI'
    #  
    #     self.ckeck_click(566.25, self.button_name)
    #     if self.normal:
    #         self.draw(surface,'white', self.button_name, 566.25)


    def button_HH(self, surface):
        self.button_name = 'H vs H'
        self.ckeck_click(566.25, self.button_name)
        if self.normal:
            self.draw(surface,'white', self.button_name,566.25)


    # def button_start(self, surface):
    #     self.draw(surface,'#ffebc6', 'START', 332.5, 450)