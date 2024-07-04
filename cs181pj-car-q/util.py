from engine.const import Const
import pickle as pickle
import math
import os.path
import random
import heapq
import functools

def saveTransProb(transDict, transFile):
    pickle.dump(transDict, transFile)

def loadTransProb():
    transFileName = Const.WORLD + 'TransProb.p'
    transFilePath = os.path.join('cs181pj-car-q','learned', transFileName)
    with open(transFilePath, "rb") as transFile:
        return pickle.load(transFile)

def xToCol(x):
    return int((x / Const.BELIEF_TILE_SIZE))

def yToRow(y):
    return int((y / Const.BELIEF_TILE_SIZE))

def rowToY(row):
    return (row + 0.5) * Const.BELIEF_TILE_SIZE

def colToX(col):
    return (col + 0.5) * Const.BELIEF_TILE_SIZE

def pdf(mean, std, value):
    u = float(value - mean) / abs(std)
    y = (1.0 / (math.sqrt(2 * math.pi) * abs(std))) * math.exp(-u * u / 2.0)
    return y

def weightedRandomChoice(weightDict):
    weights = []
    elems = []
    for elem in sorted(weightDict):
        weights.append(weightDict[elem])
        elems.append(elem)
    total = sum(weights)
    key = random.uniform(0, total)
    runningTotal = 0.0
    chosenIndex = None
    for i in range(len(weights)):
        weight = weights[i]
        runningTotal += weight
        if runningTotal > key:
            chosenIndex = i
            return elems[chosenIndex]
    raise Exception('Should not reach here')


class Belief(object):

    def __init__(self, numRows, numCols, value=None):
        self.numRows = numRows
        self.numCols = numCols
        numElems = numRows * numCols
        if value == None:
            value = (1.0 / numElems)
        self.grid = [[value for _ in range(numCols)] for _ in range(numRows)]

    def setProb(self, row, col, p):
        self.grid[row][col] = p

    def addProb(self, row, col, delta):
        self.grid[row][col] += delta
        assert self.grid[row][col] >= 0.0

    def getProb(self, row, col):
        if row >= self.numRows or col >= self.numCols:
            return 0
        return self.grid[row][col]

    def normalize(self):
        total = self.getSum()
        for r in range(self.numRows):
            for c in range(self.numCols):
                self.grid[r][c] /= total

    def getNumRows(self):
        return self.numRows

    def getNumCols(self):
        return self.numCols

    def getSum(self):
        total = 0.0
        for r in range(self.numRows):
            for c in range(self.numCols):
                total += self.getProb(r, c)
        return total


class PriorityQueue:

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)


class Counter(dict):

    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def incrementAll(self, keys, count):
        for key in keys:
            self[key] += count

    def argMax(self):
        if len(list(self.keys())) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def sortedKeys(self):
        sortedItems = list(self.items())

        def compare(x, y): return sign(y[1] - x[1])
        sortedItems.sort(key=functools.cmp_to_key(compare))
        return [x[0] for x in sortedItems]

    def totalCount(self):
        return sum(self.values())

    def normalize(self):
        total = float(self.totalCount())
        if total == 0:
            return
        for key in list(self.keys()):
            self[key] = self[key] / total

    def divideAll(self, divisor):
        divisor = float(divisor)
        for key in self:
            self[key] /= divisor

    def copy(self):
        return Counter(dict.copy(self))

    def __mul__(self, y):
        sum = 0
        x = self
        if len(x) > len(y):
            x, y = y, x
        for key in x:
            if key not in y:
                continue
            sum += x[key] * y[key]
        return sum

    def __radd__(self, y):
        for key, value in list(y.items()):
            self[key] += value

    def __add__(self, y):
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] + y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = y[key]
        return addend

    def __sub__(self, y):
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] - y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = -1 * y[key]
        return addend


def sign(x):
    if (x >= 0):
        return 1
    else:
        return -1


def flipCoin(p):
    r = random.random()
    return r < p