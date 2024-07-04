from engine.const import Const
import util, math

class NoInference(object):

    def __init__(self, numRows, numCols):
        self.belief = util.Belief(numRows, numCols)
   
    def observe(self, agentX, agentY, observedDist):
        pass

    def elapseTime(self):
        pass

    def getBelief(self):
        return self.belief
    

