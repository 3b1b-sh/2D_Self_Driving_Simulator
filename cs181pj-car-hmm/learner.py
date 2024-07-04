import util, collections

class Learner(object):

    def __init__(self):
        self.transitions = dict() # oldTile --> counter newTile

    def noteCarMove(self, oldPos, newPos):
        oldRow, oldCol = util.yToRow(oldPos.y), util.xToCol(oldPos.x)
        newRow, newCol = util.yToRow(newPos.y), util.xToCol(newPos.x)
        oldTile = (oldRow, oldCol)
        newTile = (newRow, newCol)

        if oldTile in self.transitions:
            self.transitions[oldTile][newTile] += 1
        else:
            self.transitions[oldTile] = collections.Counter()
            self.transitions[oldTile][newTile] = 1

    def saveTransitionProb(self, transFile):
        transProb = {}
        for oldTile in self.transitions:
            counter = self.transitions[oldTile]
            s = float(sum(counter.values()))
            for key in counter:
                counter[key] /= s
        for oldTile in self.transitions:
            for newTile in self.transitions:
                transProb[(oldTile, newTile)] = self.transitions[oldTile][newTile]
        
        util.saveTransProb(transProb, transFile)
        

