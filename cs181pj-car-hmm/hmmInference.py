import collections
import random
import util
from engine.const import Const
from util import Belief

class ExactInference:

    def __init__(self, numRows: int, numCols: int):
        self.skipElapse = False
        self.belief = util.Belief(numRows, numCols)
        self.transProb = util.loadTransProb()

    def observe(self, agentX: int, agentY: int, observedDist: float) -> None:
        for row in range(self.belief.getNumRows()):
            for col in range(self.belief.getNumCols()):
                mu = ((util.rowToY(row) - agentY) ** 2 + (util.colToX(col) - agentX) ** 2) ** 0.5
                emission_prob = util.pdf(mu, Const.SONAR_STD, observedDist)
                self.belief.setProb(row, col, self.belief.getProb(row, col) * emission_prob)
        self.belief.normalize()

    def elapseTime(self) -> None:
        if self.skipElapse:
            return
        newBelief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)
        for (oldTile, newTile) in self.transProb:
            newBelief.addProb(newTile[0], newTile[1], self.transProb[(oldTile, newTile)] * self.belief.getProb(oldTile[0], oldTile[1]))
        newBelief.normalize()
        self.belief = newBelief

    def getBelief(self) -> Belief:
        return self.belief

class ParticleFilter:
    NUM_PARTICLES = 200

    def __init__(self, numRows: int, numCols: int):
        self.belief = util.Belief(numRows, numCols)

        self.transProb = util.loadTransProb()
        self.transProbDict = dict()
        for (oldTile, newTile) in self.transProb:
            if oldTile not in self.transProbDict:
                self.transProbDict[oldTile] = collections.defaultdict(int)
            self.transProbDict[oldTile][newTile] = self.transProb[(oldTile, newTile)]

        # Initialize the particles randomly.
        self.particles = collections.defaultdict(int)
        potentialParticles = list(self.transProbDict.keys())
        for _ in range(self.NUM_PARTICLES):
            particleIndex = int(random.random() * len(potentialParticles))
            self.particles[potentialParticles[particleIndex]] += 1

        self.updateBelief()

    def updateBelief(self) -> None:
        newBelief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)
        for tile in self.particles:
            newBelief.setProb(tile[0], tile[1], self.particles[tile])
        newBelief.normalize()
        self.belief = newBelief

    def observe(self, agentX: int, agentY: int, observedDist: float) -> None:
        reWeightedParticles = collections.defaultdict(int)
        for particle in self.particles:
            mu = ((util.rowToY(particle[0]) - agentY) ** 2 + (util.colToX(particle[1]) - agentX) ** 2) ** 0.5
            emission_prob = util.pdf(mu, Const.SONAR_STD, observedDist)
            reWeightedParticles[particle] = self.particles[particle] * emission_prob 
        newParticles = collections.defaultdict(int)
        for _ in range(self.NUM_PARTICLES):
            newParticles[util.weightedRandomChoice(reWeightedParticles)] += 1
        self.particles = newParticles

        self.updateBelief()

    def elapseTime(self) -> None:
        newParticles = collections.defaultdict(int)
        for particle in self.particles:
                for _ in range(self.particles[particle]):
                    newParticle = util.weightedRandomChoice(self.transProbDict[particle])
                    newParticles[newParticle] += 1
        self.particles = newParticles    

    def getBelief(self) -> Belief:
        return self.belief

