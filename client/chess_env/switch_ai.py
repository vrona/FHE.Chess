import sys
from base import *


sys.path.insert(1,"server/")
from network import Network

class AI:

    def __init__(self, mode='H vs H', ai_on=False):

        self.mode = mode
        self.ai_on = ai_on


    def switch_AI(self, mode, ai_on):
        if mode == 'WHITE AI':
            self.synthetic_player = 'white'
            self.ai_on = True
        
        elif mode == 'BLACK AI':
            self.synthetic_player = 'black'
            self.ai_on = True

        elif mode == 'AI vs AI':

class White(AI):

     def __init__(self, mode):
        if mode == 'WHITE AI':
            self.synthetic_player = 'white'
            self.ai_on = True

        super().__init__('WHITE AI', mode, True)


class Black(AI):

     def __init__(self, mode):
        self.synthetic_player = 'black' if mode == 'BLACK AI' else None
        super().__init__('BLACK AI', mode, True)


class Switch_AI:

    def __init__(self, color, ai_on=False):
        
        self.ai_on = ai_on
        self.color = color
        self.ai = False
        self.cs_network = Network()
        self.mode = 'H vs H'


    def white_AI(self, mode, ai_on):
        self.ai = True if mode == 'WHITE AI' and self.ai_on else False


    def black_AI(self, mode, ai_on):
        self.ai = True if color == 'BLACK AI' and  self.ai_on else False
        
    
    def AI_vs_AI(self, mode, ai_on):
        self.ai = True if color == 'AI vs AI' and  self.ai_on else False
        pass

    def H_vs_H(self, mode, ai_on):
        self.ai = True if color == 'H vs H' and  self.ai_on else False
        pass

    def ai_play(self, board):
        if self.mode == 
        self.cs_network.send()
