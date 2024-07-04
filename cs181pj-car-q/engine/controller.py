import copy
from .const import Const
from .view.display import Display
from .model.layout import Layout
from .vector import Vec2d
from .containers.counter import Counter
from .model.QModel import QModel
import util as util
from .view import graphicsUtils
import time
import math
import sys
import traceback


class Controller(object):

    def __init__(self, weights=None, alpha=None, epsilon=None, showGraphics=True):

        self.layout = Layout(Const.WORLD)
        self.showGraphics = showGraphics


        """QAgent initialization"""
        if showGraphics:
            Display.initGraphics(self.layout)
        self.model = QModel(self.layout, weights, alpha, epsilon)
        self.agentIndex = 1  # HMM agent by default; 1 for QAgent
        self.isLearning = True
        if alpha == 0 and epsilon == 0:
            self.isLearning = False

        self.carChanges = {}
        self.errorCounter = Counter()
        self.consecutiveLate = 0
        self.gameOver = False
        self.quit = False
        self.victory = False
        self.collided = False
    
    def learn(self, learner):
        self.isLearning = True
        self.learner = learner
        return self.run()
    
    def q_learn(self):
        # self.isLearning = True
        isquit, win, iters = self.run()
        return isquit, self.model.junior.weights, win, iters

    def drive(self):
        self.isLearning = False
        return self.run()
    
    def isQLearning(self):
        return self.isLearning and self.agentIndex == 1

    def run(self):
        if self.showGraphics:
            self.render()

        self.iteration = 0
        while not self.isGameOver():
            self.resetTimes()
            startTime = time.time()

            junior = self.model.junior
            oldDir = Vec2d(junior.dir.x, junior.dir.y)
            oldPos = Vec2d(junior.pos.x, junior.pos.y)
            quitAction = junior.action()
            # carProb = self.model.getProbCar()
            # if carProb and Const.AUTO:
            if self.agentIndex == 0:
                carProb = self.model.getProbCar()
                if carProb and Const.AUTO:
                    agentGraph = self.model.getJuniorGraph()
                    junior.autonomousAction(carProb, agentGraph)
            else:
                if Const.AUTO:
                    junior.autonomousAction(self.model)

            if quitAction:
                self.gameOver = True
                self.quit = True
                break
            junior.update()

            if self.iteration > 1000 and not self.isLearning:
                # print("Time out after ", self.iteration, " exploration iterations.")
                self.gameOver = True
                break

            if self.collided or self.victory:
                self.gameOver = True

            if self.showGraphics:
                newPos = junior.getPos()
                newDir = junior.getDir()
                deltaPos = newPos - oldPos
                deltaAngle = oldDir.get_angle_between(newDir)
                Display.move(junior, deltaPos)
                Display.rotate(junior, deltaAngle)

            self.otherCarUpdate()
            # MARK: Not actived when QAgent is learning
            if self.agentIndex == 0:
                self.calculateError()

            self.collided = self.model.checkCollision(junior)
            self.victory = self.model.checkVictory()
            if self.showGraphics:
                duration = time.time() - startTime
                timeToSleep = Const.SECONDS_PER_HEARTBEAT - duration
                self.checkLate(timeToSleep)
                timeToSleep = max(0.01, timeToSleep)
                Display.graphicsSleep(timeToSleep)
            self.iteration += 1
        # if not self.userThread.quit and not self.isLearning:
        win = True
        if not self.quit and not self.isLearning:
            self.outputGameResult()
        if self.collided:
            print("Car crashed after ", self.iteration, " exploration iterations.")
            win = False
        elif self.victory:
            print("Win after", self.iteration, " exploration iterations.")
        elif self.iteration > 1000:
            print("Time out after ", self.iteration, " exploration iterations.")
            win = False

        return self.quit, win, self.iteration

    def freezeFrame(self):
        if not self.showGraphics:
            return
        while True:
            keys = Display.getKeys()
            if 'q' in keys:
                return
            Display.graphicsSleep(0.1)

    def outputGameResult(self):
        # collided = self.userThread.hasCollided()
        if not self.showGraphics:
            return
        for car in self.model.getCars():
            Display.drawCar(car)
        print('*********************************')
        print('* GAME OVER                     *')
        if self.collided:
            print('* CAR CRASH!!!!!')
        else:
            print('* You Win!')
        print('*********************************')

    def isGameOver(self):
        if self.isLearning:
            keys = Display.getKeys()
            if 'q' in keys:
                self.quit = True
                return True
            if self.agentIndex == 0:
                return self.iteration > Const.TRAIN_ITERATIONS
            else:
                # if self.iteration > Const.TRAIN_ITERATIONS:
                #     self.model.junior.stopLearning()
                #     self.isLearning = False
                if self.victory or self.collided:
                    return True
                    # self.model.junior.setup(Vec2d(
                    #     self.model.startX, self.model.startY), self.model.startDirName, Vec2d(0, 0))
                    # print("one episode ended")
                return False
        if self.quit:
            return True
        if self.victory:
            return True
        return self.collided

    def round(self, num):
        return round(num * 1000) / 1000.0

    def checkLate(self, timeToSleep):
        secsLate = self.round(-timeToSleep)
        if secsLate > 0:
            self.consecutiveLate += 1
            if self.consecutiveLate < 3:
                return
            print('*****************************')
            print('WARNING: Late to update (' + str(secsLate) + 's)')

            print('Infer time: ' + str(self.round(self.inferTime)))
            print('Action time: ' + str(self.round(self.actionTime)))
            print('Update time: ' + str(self.round(self.updateTime)))
            print('Draw time: ' + str(self.round(self.drawTime)))
            print('*****************************')
        else:
            self.consecutiveLate = 0

    def resetTimes(self):
        self.actionTime = 0
        self.inferTime = 0
        self.drawTime = 0
        self.updateTime = 0

    def juniorUpdate(self):
        junior = self.model.junior
        junior.action()
        self.move([junior])

    def otherCarUpdate(self):
        if (True or Const.INFERENCE != 'none') and self.agentIndex == 0:
            self.infer()
        self.act()
        self.move(self.model.getOtherCars())

    def act(self):
        start = time.time()
        for car in self.model.getOtherCars():
            car.action()
        self.actionTime += time.time() - start

    def move(self, cars):
        for car in cars:
            start = time.time()
            oldDir = Vec2d(car.dir.x, car.dir.y)
            oldPos = Vec2d(car.pos.x, car.pos.y)
            car.update()
            newPos = car.getPos()
            newDir = car.getDir()
            deltaPos = newPos - oldPos
            deltaAngle = oldDir.get_angle_between(newDir)
            self.updateTime += time.time() - start
            if self.showGraphics and (Const.SHOW_CARS or car.isJunior()):
                self.moveCarDisplay(car, deltaPos, deltaAngle)

            if self.isLearning and self.agentIndex == 0:
                self.learner.noteCarMove(oldPos, newPos)

    def moveCarDisplay(self, car, deltaPos, deltaAngle):
        start = time.time()
        Display.move(car, deltaPos)
        Display.rotate(car, deltaAngle)
        self.drawTime += time.time() - start

    def render(self):
        Display.drawBelief(self.model)
        Display.drawBlocks(self.model.getBlocks())
        if Const.SHOW_CARS:
            for car in self.model.getCars():
                Display.drawCar(car)
        else:
            Display.drawCar(self.model.getJunior())
        Display.drawFinish(self.model.getFinish())
        graphicsUtils.refresh()
