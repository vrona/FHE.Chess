from piece import *
from square import Square

class Move:

    def __init__(self, source, target):
        
        self.source = source
        self.target = target

    def __eq__(self, other): # explicit definition of move equality
        return self.source == other.source and self.target == other.target