data = pd.read_csv('/content/chess_games.csv')
short_data = data.head(5)
one_match = short_data.iloc[1]
display(one_match)

Event                                                         Blitz 
White                                                         go4jas
Black                                                     Sergei1973
Result                                                           0-1
UTCDate                                                   2016.06.30
UTCTime                                                     22:00:01
WhiteElo                                                        1641
BlackElo                                                        1627
WhiteRatingDiff                                                -11.0
BlackRatingDiff                                                 12.0
ECO                                                              C20
Opening                                    King's Pawn Opening: 2.b3
TimeControl                                                    300+0
Termination                                                   Normal
AN                 1. e4 e5 2. b3 Nf6 3. Bb2 Nc6 4. Nf3 d6 5. d3 ...
Name: 1, dtype: object