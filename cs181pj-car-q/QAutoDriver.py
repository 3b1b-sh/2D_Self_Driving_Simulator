import copy
from engine.model.car.junior import Junior
from engine.model.car.car import Car
# from engine.model.QModel import QModel
from engine.vector import Vec2d
from engine.const import Const
import util
import random
import math


class QAutoDriver(Junior):
    def __init__(self, layout, weights: util.Counter, epsilon=0.4, alpha=0.5, discount=1, burnInIterations=5):
        self.nodeId = None
        self.nextId = None
        self.nextNode = None
        self.burnInIterations = burnInIterations  # Number of iterations to explore before learning
        self.frontier = util.PriorityQueue()
        self.goalNodeId = None
        self.startNodeId = None
        self.backwardCostRecord = {}
        self.weights = weights  # Weights for the feature extractor
        self.featureExtractor = self.getFeatures  # Feature extractor function
        self.isLearning = True
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha    # Learning rate
        self.discount = discount  # Discount factor

        self.world_width = layout.getWidth()
        self.world_height = layout.getHeight()
        self.world_size = self.world_width * self.world_height
    
        self.lastReward = None
        self.lastFeatures = None
        self.lastQValue = None

        self.legalActions = []

        drive_forward = [0.0, 4.0]
        turn_left = [0.0, 20.0]
        turn_right = [0.0, 20.0]
        for l in turn_left:
            for r in turn_right:
                # Check Legal: Not both left and right
                if l == 0.0 or r == 0.0:
                    for f in drive_forward:
                        action = {}
                        if f != 0.0: action[Car.DRIVE_FORWARD] = f
                        if l != 0.0: action[Car.TURN_LEFT] = l
                        if r != 0.0: action[Car.TURN_RIGHT] = r
                        self.legalActions.append(action)

    def getQValue(self, model, action):
        features = self.featureExtractor(model, action)
        # QValue = sum(self.weights[f] * value for f, value in features.items())
        QValue = 0.0
        for feature, value in features.items():
            QValue += self.weights[feature] * value

        return QValue

    def computeValueFromQValues(self, model):
        # Return the maximum Q-value among all possible actions from the current state
        possible_actions = self.getLegalActions()
        if not possible_actions:
            return 0.0
        return max(self.getQValue(model, action) for action in possible_actions)

    def computeActionFromQValues(self, model):
        # Choose the best action based on Q-values
        possible_actions = self.getLegalActions()
        if not possible_actions:
            return None

        best_actions = []
        best_q_value = float("-inf")
        for action in possible_actions:
            q_value = self.getQValue(model, action)
            if q_value > best_q_value:
                best_q_value = q_value
                best_actions = [action]
            elif q_value == best_q_value:
                best_actions.append(action)
        best_action = random.choice(best_actions)
        return best_action

    def getLegalActions(self) -> list:
        return self.legalActions
    
    def getForwardAction(self):
        return {Car.DRIVE_FORWARD: 4.0, Car.TURN_LEFT: 0.0, Car.TURN_RIGHT: 0.0}

    def updateQValue(self, features, QValue, nextState, reward):
        # Update Q-value based on the transition
        if self.isLearning:
            # print(reward)
            # features = self.featureExtractor(model, action, amount)
            correction = reward + self.discount * self.computeValueFromQValues(nextState) - QValue
            for feature, value in features.items():
                self.weights[feature] += self.alpha * correction * value

    def getAutonomousActions(self, model):
        if self.burnInIterations > 0:
            self.burnInIterations -= 1
            return None, 0

        if util.flipCoin(self.epsilon):
            return random.choice(self.getLegalActions())
        else:
            # Choose the best action based on Q-values
            return self.computeActionFromQValues(model)

    def getFeatures(self, model, action):

        tmp_car = Junior(Vec2d(model.junior.pos), "east", Vec2d(model.junior.velocity))
        tmp_car.dir = Vec2d(model.junior.dir)

        tmp_car.applyActions(action)

        tmp_car.update()

        features = util.Counter()
        features['bias'] = 1.0

        x_position = tmp_car.pos.x
        y_position = tmp_car.pos.y
        # direction = model.junior.dir.get_angle()

        closest_block_distance = float('inf')
        for block in model.blocks:
            dist = block.distance_to_point(
                x_position, y_position)/self.world_size
            if dist < closest_block_distance:
                closest_block_distance = dist

        closest_car_distance = float('inf')
        for car in model.getCars():
            if car != model.junior:
                car_pos = car.pos
                dist = math.sqrt((car_pos.x - x_position) ** 2 +
                                 (car_pos.y - y_position) ** 2)/self.world_size
                if dist < closest_car_distance:
                    closest_car_distance = dist

        finish_pos = model.getFinish().getCenter()
        distance_to_finish = (math.sqrt(
            (finish_pos.x - x_position) ** 2 + (finish_pos.y - y_position) ** 2)) / self.world_size
        
        nearest_boundary_distance = min(x_position, y_position, self.world_width - x_position, self.world_height - y_position) / (self.world_width * self.world_height)

        # features['x_position'] = x_position
        # features['y_position'] = y_position
        features['distance_to_finish'] = distance_to_finish
        features['closest_block_distance'] = closest_block_distance
        features['closest_car_distance'] = closest_car_distance
        features['velocity'] = tmp_car.velocity.get_length()
        features['nearest_boundary_distance'] = nearest_boundary_distance
        car_to_final = model.finish.getCenter() - tmp_car.getPos()
        # features['velocity_to_finish'] = tmp_car.velocity.dot(car_to_final.normalized())
        if tmp_car.velocity.dot(car_to_final.normalized()) > 0:
            features['move_towards_finish'] = 1
        else:
            features['move_towards_finish'] = 0
        
        return features

    def autonomousAction(self, model):
        action = self.getAutonomousActions(model)
        if action is None:
            return

        # print("Selected action: ", action, amount)

        if Car.DRIVE_FORWARD in action:
            percent = action[Car.DRIVE_FORWARD]
            percent = max(percent, 0.0)
            percent = min(percent, 1.0)
            self.accelerate(Junior.ACCELERATION * percent)
        if Car.TURN_LEFT in action:
            self.turnLeft(action[Car.TURN_LEFT])
        if Car.TURN_RIGHT in action:
            self.turnRight(action[Car.TURN_RIGHT])

        if self.lastReward and self.isLearning:
            self.updateQValue(self.lastFeatures, self.lastQValue, 
                              model, self.lastReward)

        self.lastFeatures = self.featureExtractor(model, action)
        self.lastReward = model.getReward(action)
        self.lastQValue = self.getQValue(model, action)
