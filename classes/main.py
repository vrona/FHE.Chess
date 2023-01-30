import pygame
import sys

from base import *
from game import Game

class Main:

    def __init__(self):
        pygame.init()
        self.screenplay = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Zama FHE.Chess')
        self.game = Game()

    def mainloop(self):
        
        while True:
            self.game.show_backg(self.screenplay)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()




            pygame.display.update()


main = Main()
main.mainloop()