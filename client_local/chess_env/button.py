import pygame
from PIL import ImageFont
from base import *

class Button:
	
    def __init__(self, normal = True, y = sp_height/2 -30):

        self.normal = normal
        self.y_pos = y
        self.ai_mode = False
        self.name_mode = ' White H'
        self.restart = True

    # retrieve if AI Mode status
    def get_ai_mode(self):
        return self.ai_mode
    
    # click function
    def check_click(self, x, action_name):

        mouse_pos = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()[0]
        button_rect = pygame.rect.Rect((x, self.y_pos),(130,30))
        
        if click and button_rect.collidepoint(mouse_pos) and self.normal:
            # set the AI or Human mode
            self.name_mode = action_name
            self.normal = False

            if self.name_mode=="White AI ":
                self.ai_mode = True

            if self.restart:
                self.normal = True
                self.restart = False

        else:
           return False

    # get in pixel the size of text
    """ warning ImageFont calls the same font (from client_local/content/) as PyGame (from site-packages/pygame/)"""
    def sizeoftext(self, text):
        font = ImageFont.truetype('client_local/content/FreeSans/FreeSansBold.ttf', 28)
        text_width = font.getlength(self.text)
        return text_width

    # draw button
    def draw(self, surface, color, text, x, y = 0):

        self.text = text
        text_width = self.sizeoftext(self.text)

        font = pygame.font.Font('freesansbold.ttf', 26)
        button_text = font.render(self.text, True, 'black')
        button_rect = pygame.rect.Rect((x, self.y_pos + y),(text_width,30))
        pygame.draw.rect(surface, color, button_rect, 0, 5)
        pygame.draw.rect(surface, 'black', button_rect, 2, 3)
        surface.blit(button_text, (x+3, self.y_pos + y +3))


    def button_whiteAI(self, surface):
        self.button_name = 'White AI '
        self.check_click(220, self.button_name)
        if self.normal:
            self.draw(surface,'white', self.button_name,220)

    def button_HH(self, surface):
        self.button_name = 'White Human'
        self.check_click(445, self.button_name)
        if self.normal:
            self.draw(surface,'white', self.button_name,445)

    def button_restart(self, surface):
        self.button_name = 'New Game'
        self.check_click(445, self.button_name)
        if self.restart:
            self.draw(surface,'white', self.button_name,332)

    def show_result(self, surface, winner, termination):
        self.draw(surface, '#DE3163', "%s" % winner, 1, -340) # winner or draw
        self.draw(surface, '#DE3163', "%s" % termination, 1, -305) # reason

    # def button_blackAI(self, surface):
    #     self.button_name = 'BLACK AI'
    #     self.check_click(332.5, self.button_name)
    #     if self.normal:
    #         self.draw(surface, 'white', self.button_name, 332.5)

    # def button_bothAI(self, surface):
    #     self.button_name = 'AI vs AI'
    #     self.check_click(566.25, self.button_name)
    #     if self.normal:
    #         self.draw(surface,'white', self.button_name, 566.25)
