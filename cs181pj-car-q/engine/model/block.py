from engine.vector import Vec2d
from engine.const import Const
import math


class Block(object):

    def __init__(self, row):
        self.x1 = row[0] * Const.BLOCK_TILE_SIZE
        self.y1 = row[1] * Const.BLOCK_TILE_SIZE
        self.x2 = row[2] * Const.BLOCK_TILE_SIZE
        self.y2 = row[3] * Const.BLOCK_TILE_SIZE
        self.centerX = (self.x1 + self.x2) / 2.0
        self.centerY = (self.y1 + self.y2) / 2.0

    def getCenter(self):
        return Vec2d(self.centerX, self.centerY)

    def getWidth(self):
        return abs(self.x2 - self.x1)

    def getHeight(self):
        return abs(self.y2 - self.y1)

    def containsPoint(self, x, y):
        if x < self.x1:
            return False
        if y < self.y1:
            return False
        if x > self.x2:
            return False
        if y > self.y2:
            return False
        return True

    def distance_to_point(self, x, y):
        if x < self.x1:
            dx = self.x1 - x
        elif x > self.x2:
            dx = x - self.x2
        else:
            dx = 0

        if y < self.y1:
            dy = self.y1 - y
        elif y > self.y2:
            dy = y - self.y2
        else:
            dy = 0

        return math.sqrt(dx**2 + dy**2)
