from .car.car import Car
from .car.agent import Agent
from .car.junior import Junior
from engine.vector import Vec2d
from engine.const import Const
from engine.model.block import Block
from engine.model.agentCommunication import AgentCommunication
from QAutoDriver import QAutoDriver

import threading
import copy
import util


class QModel(object):
    '''
    Class: QModel    (test only)
    World Information: 
    QAutoDriver
    IsEnd(s): checkVictory or checkCollision
    Actions(s): in the QAutoDriver class (parent class Car) -> (DRIVE_FORWARD, TURN_LEFT, TURN_RIGHT, TURN_WHEEL) by fixed amount, (copy the constant from Junior Class) 
    Reward(s, a, s'): 
        1. Arrive destination: +1000 (checkVictory)
        2. Collision: -1000 (checkCollision)
        3. Move: -1 (no collision or victory)
    Transition(s, a, s'): 1 (if s' is the desired destination), 0 (otherwise)
    '''

    def __init__(self, layout, weights, alpha=0.5, epsilon=0.5):
        self._initBlocks(layout)
        self._initIntersections(layout)
        self.layout = layout
        self.startX = layout.getStartX()
        self.startY = layout.getStartY()
        self.startDirName = layout.getJuniorDir()
        self.junior = QAutoDriver(layout, weights, alpha, epsilon)
        self.junior.setup(
            Vec2d(self.startX, self.startY),
            self.startDirName,
            Vec2d(0, 0)
        )
        self.cars = [self.junior]
        self.otherCars = []
        self.finish = Block(layout.getFinish())
        agentComm = AgentCommunication()
        agentGraph = layout.getAgentGraph()
        for _ in range(Const.NUM_AGENTS):
            startNode = self._getStartNode(agentGraph)
            other = Agent(startNode, layout.getAgentGraph(), self, agentComm)
            self.cars.append(other)
            self.otherCars.append(other)
        self.observations = []
        agentComm.addAgents(self.otherCars)
        # self.modelLock = threading.Lock()
        self.probCarSet = False

    def _initBlocks(self, layout):
        self.blocks = []
        for blockData in layout.getBlockData():
            block = Block(blockData)
            self.blocks.append(block)

    def _initIntersections(self, layout):
        self.intersections = []
        for blockData in layout.getIntersectionNodes():
            block = Block(blockData)
            self.intersections.append(block)

    def _getStartNode(self, agentGraph):
        while True:
            node = agentGraph.getRandomNode()
            pos = node.getPos()
            alreadyChosen = False
            for car in self.otherCars:
                if car.getPos() == pos:
                    alreadyChosen = True
                    break
            if not alreadyChosen:
                return node

    def checkVictory(self):
        bounds = self.junior.getBounds()
        for point in bounds:
            if self.finish.containsPoint(point.x, point.y):
                return True
        return False

    def checkCollision(self, car):
        bounds = car.getBounds()
        # check for collision with fixed obstacles
        for point in bounds:
            if not self.inBounds(point.x, point.y):
                return True

        # check for collision with other cars
        for other in self.otherCars:
            if other.collides(car.getPos(), bounds):
                return True
        return False

    def getIntersection(self, x, y):
        for intersection in self.intersections:
            if intersection.containsPoint(x, y):
                return intersection
        return None

    def inIntersection(self, x, y):
        return self.getIntersection(x, y) != None

    def inBounds(self, x, y):
        if x < 0 or x >= self.getWidth():
            return False
        if y < 0 or y >= self.getHeight():
            return False
        for block in self.blocks:
            if block.containsPoint(x, y):
                return False
        return True

    def getWidth(self):
        return self.layout.getWidth()

    def getHeight(self):
        return self.layout.getHeight()

    def getBeliefRows(self):
        return self.layout.getBeliefRows()

    def getBeliefCols(self):
        return self.layout.getBeliefCols()

    def getBlocks(self):
        return self.blocks

    def getFinish(self):
        return self.finish

    def getCars(self):
        return self.cars

    def getOtherCars(self):
        return self.otherCars

    def getJunior(self):
        return self.junior

    def getAgentGraph(self):
        return self.layout.getAgentGraph()

    def getJuniorGraph(self):
        return self.layout.getJuniorGraph()

    def getReward(self, action):
        # futureModel = (self)
        # actions = {action: amount}
        # futureModel.junior.applyActions(actions)
        # futureModel.junior.update()
        # if futureModel.checkVictory():
        #     return 1000
        # if futureModel.checkCollision(futureModel.junior):
        #     return -500
        # if action == Car.DRIVE_FORWARD:
        #     return 10
        # elif action in [Car.TURN_LEFT, Car.TURN_RIGHT]:
        #     return -1
        # return -0.1
        tmp_junior = Junior(Vec2d(self.junior.getPos()),
                            "north", Vec2d(self.junior.velocity))
        tmp_junior.dir = Vec2d(self.junior.getDir())
        saved_junior = self.junior
        self.junior = tmp_junior
        self.junior.applyActions(action)
        self.junior.update()
        reward = -1
        if self.checkVictory():
            reward = 100000
        # out of bound
        elif self.checkCollision(self.junior) or self.junior.getPos().x <= 0 or self.junior.getPos().x >= self.getWidth() or self.junior.getPos().y <= 0 or self.junior.getPos().y >= self.getHeight():
            reward = -1000
        elif Car.DRIVE_FORWARD in action:
            reward = 20

        if Car.TURN_LEFT in action or Car.TURN_RIGHT in action:
            reward -= 5

        reward -= 15

        # bonus of velocity
        reward += self.junior.velocity.get_length() * 10

        reward += self.junior.getPos().x * 0.8

        car_to_final = self.finish.getCenter() - self.junior.getPos()
        # rejection of car.velocity to dir of car_to_final
        reward += self.junior.velocity.dot(car_to_final.normalized()) * 50

        self.junior = saved_junior
        return reward
    # def isActionSafe(self, action, amount):
    #     # Simulate the action and check if it results in a collision
    #     test_junior = copy.deepcopy(self.junior)  # Create a copy of the junior
    #     test_junior.applyActions(action)  # Apply the action to the copy
    #     # Check if the action results in a collision
    #     return not test_junior.checkCollision()
