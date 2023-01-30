import pygame
import sys

from base import *
from game import Game

class Main:

    def __init__(self):
        pygame.init()
        self.screenplay = pygame.display.set_mode((sp_width, sp_height))
        pygame.display.set_caption('Zama FHE Chess')
        self.game = Game()

    def mainloop(self):
        
        screenplay = self.screenplay
        game = self.game

        while True:
            game.show_backg(screenplay)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()




            pygame.display.update()


main = Main()
main.mainloop()