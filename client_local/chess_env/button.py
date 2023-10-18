import pygame
from PIL import ImageFont
from base import *

class Button:

    def __init__(self, normal = True, restart = False, y = sp_height/2 -30):

        self.normal = normal
        self.y_pos = y
        self.white_ai = False
        self.black_ai = False
        self.white_human = False
        self.black_human = False
        self.name_mode = None
        self.new_game = False
        self.click_new = None
        self.restart = restart

    # retrieve if AI Mode status   
    def is_white_ai_(self):
        return self.white_ai
    
    def is_black_ai_(self):
        return self.black_ai

    def is_color_human_(self, color):
        if color=="white":
            return self.white_human
        if color=="black":
            return self.black_human
    
    # click function
    def click_ai(self, x, action_name, network):
        """
        x: int initial pixel within width
        action_name: string White AI or White Human
        network: bool
        """
        text_width = self.sizeoftext(action_name)
        mouse_pos = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()[0]
        button_rect = pygame.rect.Rect((x, self.y_pos),(text_width,30))
        
        if click and button_rect.collidepoint(mouse_pos) and self.normal:
            
            if not network.connected:
                network.input_ip(network.connected)
                self.click_ai(x, action_name, network)

            else:
                # set the AI mode
                self.name_mode = action_name
                self.normal = False
                self.restart = True

                if self.name_mode==" White AI ":
                    self.white_ai = True
                    self.black_human = True

                if self.name_mode==" Black AI":
                    self.black_ai = True
                    self.white_human = True

                if self.name_mode=="AI vs AI":
                    self.white_ai = True
                    self.black_ai = True
            
                if self.new_game:
                    self.new_game = False
    
    def click_human(self, x, action_name):
        text_width = self.sizeoftext(action_name)
        mouse_pos = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()[0]
        button_rect = pygame.rect.Rect((x, self.y_pos),(text_width,30))
        
        if click and button_rect.collidepoint(mouse_pos) and self.normal:
            # set the Human mode
            self.name_mode = action_name
            self.normal = False
            self.restart = True

            if self.name_mode=="H vs H":
                self.white_human = True
                self.black_human = True

            if self.new_game:
                self.new_game = False

    def click_new_game(self, x, action_name):
        text_width = self.sizeoftext(action_name)
        mouse_pos = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()[0]
        button_rect = pygame.rect.Rect((x, self.y_pos),(text_width,30))
        
        if click and button_rect.collidepoint(mouse_pos) and self.restart:
            # set the new game mode
            self.click_new = click
            self.restart= False
            self.new_game = True

    # get in pixel the size of text
    def sizeoftext(self, text):
        """
        warning:
        ImageFont calls the same font (from client_local/content/) as PyGame (from site-packages/pygame/)
        """
        font = ImageFont.truetype('client_local/content/FreeSans/FreeSansBold.ttf', 30)
        text_width = font.getlength(text)
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

    def button_whiteAI(self, surface, bool):
        self.button_name = ' White AI '
        self.click_ai(220, self.button_name, bool)
        if self.normal:
            self.draw(surface,'white', self.button_name,220)

    def button_blackAI(self, surface, bool):
        self.button_name = ' Black AI'
        self.click_ai(445, self.button_name, bool)
        if self.normal:
            self.draw(surface, 'white', self.button_name, 445)

    def button_bothAI(self, surface, bool):
        self.button_name = 'AI vs AI'
        self.click_ai(691, self.button_name,bool)
        if self.normal:
            self.draw(surface,'white', self.button_name, 691)

    def button_dev(self, surface):
        self.button_name = 'H vs H'
        self.click_human(0, self.button_name)
        if self.normal:
            self.draw(surface,'white', self.button_name,0)

    def button_restart(self, surface):
        self.button_name = 'New Game'
        self.click_new_game(332, self.button_name)
        if self.restart:
            self.draw(surface,'white', self.button_name,332)
        return True

    def show_result(self, surface, winner, termination):
        if self.restart:
            self.draw(surface, '#ededed', "%s" % winner, 98, -240) # winner or draw
            self.draw(surface, '#ededed', "%s" % termination, 98, -205) # reason

    def show_AI_givingup(self, surface, text_AI_go):
        if self.restart:
            self.draw(surface, '#ff7400', "%s" % text_AI_go, 98, -170)