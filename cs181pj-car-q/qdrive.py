from engine.controller import Controller
from engine.const import Const
from engine.view.display import Display

import sys
import optparse
import random
import signal
import util
import numpy as np
import matplotlib.pyplot as plt


def signal_handler(signal, frame):
    Display.raiseEndGraphics()


def drive(weights, showGraphics):
    alpha = 0
    epsilon = 0
    showGraphics = showGraphics
    NUM = 100
    scores = []
    wins = 0
    loses = 0
    for i in range(NUM):
        controller = Controller(weights, alpha, epsilon, showGraphics)
        quit, _, win, iters = controller.q_learn()
        if quit:
            break
        if i > 10:
            showGraphics = False
        if win:
            wins += 1
            scores.append(iters)
        else:
            loses += 1
    # plot the scores
    # bins = np.arange(80, 221, 20)
    # bins = np.append(bins, np.inf)

    # hist, bin_edges = np.histogram(scores, bins=bins)

    # plt.bar(range(len(hist)), hist, tick_label=[f"{int(bin_edges[i])}-{int(bin_edges[i+1])-1}" if i < len(bin_edges) - 2 else f"{int(bin_edges[i])}+" for i in range(len(bin_edges) - 1)])
    # plt.xlabel('Step before reaching the finish')
    # plt.ylabel('Frequency')
    # plt.title('Step Distribution')
    # plt.show()
    print("Done driving...")
    print("With weights: ", weights)

    # # calculate the score statistics
    if len(scores) == 0:
        print("No effective score")
    else:
        print("Average score: ", np.mean(scores))
        print("Max score: ", np.max(scores))
        print("Min score: ", np.min(scores))
        print("Standard deviation: ", np.std(scores))

    print("Wins: ", wins, " Loses: ", loses)


if __name__ == '__main__':

    parser = optparse.OptionParser()
    parser.add_option('-k', '--numCars', type='int', dest='numCars', default=3)
    parser.add_option('-l', '--layout', dest='layout', default='small')
    parser.add_option('-f', '--fixedSeed', dest='fixedSeed',
                      default=False, action='store_true')
    parser.add_option('-t', '--train', dest='train',
                      default=False, action='store_true')
    parser.add_option('-g', '--showGraph', dest='showGraph',
                      default=False, action='store_true')
    parser.add_option('-i', '--numIter', dest='numIter',
                      type='int', default=100)
    (options, _) = parser.parse_args()

    Const.WORLD = options.layout
    Const.CARS_PARKED = False
    Const.SHOW_CARS = True
    Const.NUM_AGENTS = options.numCars
    Const.INFERENCE = 'exactInference'
    # Const.SPEED = verySlow
    Const.HEARTBEATS_PER_SECOND = Const.HEARTBEAT_DICT["veryFast"]
    Const.SECONDS_PER_HEARTBEAT = 1.0 / Const.HEARTBEATS_PER_SECOND
    Const.AUTO = True
    LEARN_NUM = options.numIter

    signal.signal(signal.SIGINT, signal_handler)

    # Fix the random seed
    if options.fixedSeed:
        random.seed('driverlessCar')

    # drive()
    # sys.exit(0)
    if options.train:

        weights = util.Counter()

        alpha_initial = 0.5
        alpha_min = 0.001
        alpha_decay_rate = 0.97

        epsilon_initial = 0.5
        epsilon_min = 0.08
        epsilon_decay_rate = 0.98

        alpha = alpha_initial
        epsilon = epsilon_initial

        showGraphics = options.showGraph

        for i in range(LEARN_NUM):
            print("Learning iteration ", i)
            controller = Controller(weights, alpha, epsilon, showGraphics)
            quit, weights, _, _ = controller.q_learn()
            print("Learned weights: ", weights)
            if quit:
                break
            alpha = max(alpha * alpha_decay_rate, alpha_min)
            epsilon = max(epsilon * epsilon_decay_rate, epsilon_min)
            # showGraphics = False
            if i > 3:
                showGraphics = False
            if i > LEARN_NUM - 5:
                showGraphics = options.showGraph
    else:
        # trained
        weights = {'bias': 919.8121383769599, 'distance_to_finish': -693.3493764881839, 'closest_block_distance': 470.56061346307487,
                   'closest_car_distance': 580.19437359727827, 'velocity': 2.4897180655117, 'nearest_boundary_distance': 153.0366614993948, 'move_towards_finish': 54.01935123373908}

    # end of training
    print("Start driving...")
    drive(weights, options.showGraph)

    print('closing...')
    Display.endGraphics()
