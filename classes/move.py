

class Move:

    def __init__(self, initial, destination):
        
        self.initial = initial
        self.destination = destination
    
    def __eq__(self, other): # explicit definition of move equality
        return self.initial == other.initial and self.destination == other.destination