
from peewee import *

db = SqliteDatabase("/Volumes/vrona_SSD/lichess_data/chess_wb2000_db.db")


class Games(Model):
    id = IntegerField()
    AN = TextField()

    class Meta:
        database = db
        #db_table = 'games'



def getid(idx):
    an = Games.get(Games.id == idx)
    
    print(an.AN)
   

    
#SQL: SELECT "t1"."id", "t1"."AN" FROM "Games" AS "t1" WHERE ("t1"."AN" = ?) LIMIT ? OFFSET ?
db.connect()

getid(504)
# for x in an:
#     print(x.id, x.AN)
db.close()

# 