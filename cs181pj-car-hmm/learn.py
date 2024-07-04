import sys
from engine.controller import Controller
from engine.const import Const
from engine.view.display import Display
from learner import Learner
import os.path
import optparse
import signal

def signal_handler(signal, frame):
    Display.raiseEndGraphics()

def run():
    parser = optparse.OptionParser()
    parser.add_option('-p', '--parked', dest='parked', default=False, action='store_true')
    parser.add_option('-d', '--display', dest='display', default=True, action='store_true')
    parser.add_option('-k', '--numCars', type='int', dest='numCars', default=3)
    parser.add_option('-l', '--layout', dest='layout', default='small')
    parser.add_option('-s', '--speed', dest='speed', default='slow')
    parser.add_option('-f', '--fixedSeed', dest='fixedSeed', default=False, action='store_true')
    parser.add_option('-a', '--auto', dest='auto', default=False, action='store_true')
    (options, _) = parser.parse_args()
    
    Const.WORLD = options.layout
    Const.CARS_PARKED = options.parked
    Const.SHOW_CARS = options.display
    Const.NUM_AGENTS = options.numCars
    Const.INFERENCE = 'none'
    Const.AUTO = options.auto
    Const.SECONDS_PER_HEARTBEAT = 0.001

    signal.signal(signal.SIGINT, signal_handler)
    
    # Fix the random seed
    if options.fixedSeed: random.seed('driverlessCar')
    
    learner = Learner()
    
    iterations = 0
    numIter = Const.TRAIN_MAX_AGENTS * Const.TRAIN_PER_AGENT_COUNT
    for i in range(1, Const.TRAIN_MAX_AGENTS + 1):
        for j in range(Const.TRAIN_PER_AGENT_COUNT):
            percentDone = int(iterations * 100.0 / numIter)
            print((str(percentDone) + '% done'))
            Const.NUM_AGENTS = i
            quit = Controller().learn(learner)
            if quit:
                Display.endGraphics()
                return
            iterations += 1
    
    transFileName = Const.WORLD + 'TransProb.p'
    transFilePath = os.path.join('cs181pj-car-hmm','learned', transFileName)
    with open(transFilePath, 'wb') as transFile:
        learner.saveTransitionProb(transFile)
        print(('saved file: ' + transFilePath))
    Display.endGraphics()

if __name__ == '__main__':
    run()
    
    
    

