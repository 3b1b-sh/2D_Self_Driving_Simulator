from engine.model.car.junior import Junior
from engine.model.car.car import Car

import util
import random

class AutoDriver(Junior):
    # Car driven by HMM Inference.

    MIN_PROB = 0.02

    def __init__(self):
        self.nodeId = None
        self.nextId = None
        self.nextNode = None
        self.burnInIterations = 30
        self.frontier = util.PriorityQueue()
        self.goalNodeId = None
        self.startNodeId = None
        self.backwardCostRecord = {}
    
    def getAutonomousActions(self, beliefOfOtherCars, agentGraph):
        if self.burnInIterations > 0:
            self.burnInIterations -= 1
            return[]
        
        if self.nodeId == None:
            self.nodeId = agentGraph.getNearestNode(self.pos)

        if self.frontier.isEmpty():
            for nodeId in agentGraph.nodeMap:
                self.frontier.push(nodeId, float("inf"))
                self.backwardCostRecord[nodeId] = float("inf")
                if agentGraph.getNode(nodeId).isTerminal():
                    self.goalNodeId = nodeId
            self.frontier.update(self.nodeId, 0)
            self.backwardCostRecord[self.nodeId] = 0
            self.startNodeId = self.nodeId
            print("Priority Queue Initialized Successfully!") 

        if self.nextId == None:
            self.choseNextId(agentGraph)
        if agentGraph.atNode(self.nextId, self.pos):
            self.nodeId = self.nextId
            self.choseNextId(agentGraph)

        goalPos = agentGraph.getNode(self.nextId).getPos()
        vectorToGoal = goalPos - self.pos
        wheelAngle = -vectorToGoal.get_angle_between(self.dir)
        driveForward = not self.isCloseToOtherCar(beliefOfOtherCars)
        actions = {
            Car.TURN_WHEEL: wheelAngle
        }
        if driveForward:
            actions[Car.DRIVE_FORWARD] = 1.0
        return actions

    def isCloseToOtherCar(self, beliefOfOtherCars):
        offset = self.dir.normalized() * 1.5 * Car.LENGTH
        newPos = self.pos + offset
        row = util.yToRow(newPos.y)
        col = util.xToCol(newPos.x)
        p = beliefOfOtherCars.getProb(row, col)
        return p > AutoDriver.MIN_PROB
 
    def choseNextId(self, agentGraph, beliefOfOtherCars=None):
        nextIds = agentGraph.getNextNodeIds(self.nodeId)
        if nextIds == []: 
            self.nextId = None
        else:
            for i in nextIds:
                nextNode = agentGraph.getNode(i)
                row = util.yToRow(nextNode.y)
                col = util.xToCol(nextNode.x)
                risk = 1
                if beliefOfOtherCars:
                    # Take the risk of a car being in a new position into account
                    risk += beliefOfOtherCars.getProb(row, col)

                self.backwardCostRecord[i] = min(self.backwardCostRecord[i], risk + self.backwardCostRecord[self.nodeId])
                goalNode = agentGraph.getNode(self.goalNodeId)
                backWardCost = self.backwardCostRecord[i]
                forwardCost = nextNode.getDist(goalNode)
                self.frontier.update(i, backWardCost+forwardCost)
            self.nextId = self.frontier.pop()

