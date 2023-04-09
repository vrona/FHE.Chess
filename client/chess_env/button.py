import pygame
from base import *

class Button:
	
    def __init__(self, y = sp_height/2 -30):

        self.normal = True
        self.y_pos = y
        self.ai_mode = False
        self.name_mode = ' White H'

    # retrieve if AI Mode status
    def get_ai_mode(self):
        return self.ai_mode
    
    # click function
    def ckeck_click(self, x, action_name):

        mouse_pos = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()[0]
        button_rect = pygame.rect.Rect((x, self.y_pos),(130,30))
        
        if click and button_rect.collidepoint(mouse_pos) and self.normal:
            # set the AI or Human mode
            self.name_mode = action_name
            self.normal = False

            if self.name_mode=="White AI":
                self.ai_mode = True

        else:
           return False

    # draw button
    def draw(self, surface, color, text, x):

        self.text = text
        font = pygame.font.Font('freesansbold.ttf', 26)
        button_text = font.render(self.text, True, 'black')
        button_rect = pygame.rect.Rect((x, self.y_pos),(120,30))
        pygame.draw.rect(surface, color, button_rect, 0, 5)
        pygame.draw.rect(surface, 'black', button_rect, 2, 3)
        surface.blit(button_text, (x+3, self.y_pos +3))


    def button_whiteAI(self, surface):
        self.button_name = 'White AI'
        self.ckeck_click(220, self.button_name)
        if self.normal:
            self.draw(surface,'white', self.button_name,220)


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
        self.button_name = ' White H'
        self.ckeck_click(445, self.button_name)
        if self.normal:
            self.draw(surface,'white', self.button_name,445)


    # def button_start(self, surface):
    #     self.draw(surface,'#ffebc6', 'START', 332.5, 450)