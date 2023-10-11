"""
Analysis chosen move in 1st place

pseudo code

Alpha Prune
if Type(Child) = Min # minimizer:
    if Uncle(Child) and Weight(Uncle):
        if Weight(Child) <= Weight(Uncle):
            Pruning
            Weight(Dad) = Weight(Child)
        
Beta Prune
if Type(Child) = Max # maximizer:
    if Uncle(Child) and Weight(Uncle):
        if Weight(Child) >= Weight(Uncle):
            Pruning
            Weight(Dad) = Weight(Child)


TRI MEMORY PRUNE_MOVE GOOD_MOVE_SEQUENCE PATTERNS
Depth 1: Heuristic sorting
Depth 2: Heuristic sorting
Depth 3: Heuristic sorting

"""