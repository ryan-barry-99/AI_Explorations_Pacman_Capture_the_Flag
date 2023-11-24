# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import json

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'QLearningCaptureAgent', second = 'QLearningCaptureAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########
         
class QLearningCaptureAgent(CaptureAgent): 
  def __init__(self, index, epsilon=0.5, alpha=0.2, discount=0.8):
    CaptureAgent.__init__(self, index)
    # Initialize variables
    self.epsilon = epsilon  # Exploration rate
    self.alpha = alpha      # Learning rate
    self.discount = discount# Discount factor
    self.weights = json.load(open('weights.json', 'r'))
    # Initialize Q-values dictionary
    self.qValues = util.Counter()
  
  def chooseAction(self, gameState):
    # Get legal actions
    actions = gameState.getLegalActions(self.index)
    # Choose action using epsilon-greedy policy
    if util.flipCoin(self.epsilon):
      # Explore: choose random action
      return random.choice(actions)
    else:
      # Exploit: choose action with highest Q-value
      maxq, best_action = self.getMaxQ(gameState)
      return best_action
  
  def getFeatures(self, gameState, action):
    self.features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    # Calculate feature values based on the state-action pair
    self.features['closest-food'] = self.calculate_closest_food(gameState)/10
    self.features['bias'] = 1.0/10
    self.features['#-of-ghosts-1-step-away'] = self.calculate_ghosts_1_step_away(gameState, action)/10
    self.features['successorScore'] = self.getScore(successor)/10
    self.features['eats-food'] = self.calculate_eats_food(gameState, action)/10
    return self.features
  
  
  def getQValue(self, gameState, action): 
    """
    Calculate the Q values by multiplying training weights with features
    Features include closest food, bias, and ghoset distance
    """
    # return self.getFeatures(gameState, action) * self.weights
    return sum(self.getFeatures(gameState, action)[f] * self.weights[f] for f in self.getFeatures(gameState, action))  
 
 
  def update(self, gameState, action, nextState, reward):
    """
    Update Q-values after taking action and observing result
    """
    self.getFeatures(gameState, action)
    self.qValues[(gameState, action)] = self.qValues[(gameState, action)] + self.alpha * (reward + self.discount * self.getMaxQ(nextState) - self.qValues[(gameState, action)])
    # self.qValues[(gameState, action)] = (1 - self.alpha) * self.getQValue(gameState, action) + self.alpha * (reward + self.discount * max(self.qValues[nextState]))
    save_weights()

  def getMaxQ(self, gameState):
    """
    Get maximum Q-value for current state
    """
    legalActions = gameState.getLegalActions(self.index)
    if not legalActions:
      return None
    else:
      qvals = []
      maxq = float('-inf')
      best_action = None
      for action in legalActions:
        qval = self.getQValue(gameState, action)
        qvals.append(qval)
        if qval > maxq:
          maxq = qval
          best_action = action
      # self.updateWeights(gameState, best_action)
        
    return maxq, best_action
  
  def getReward(self, gameState, action):
      """
      Get reward for the action at a given state
      """
      successor = self.getSuccessor(gameState, action)
      currentPosition = gameState.getAgentPosition(self.index)
      nextPosition = successor.getAgentPosition(self.index)

      reward = 0

      # Negative reward for being within a maze distance of 10 from the starting position
      if self.getMazeDistance(currentPosition, self.startPosition) <= 10:
          reward -= 999

      # Positive reward for eating food
      if self.getFood(gameState)[int(nextPosition[0])][int(nextPosition[1])]:
          reward += 1

      return reward

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != util.nearestPoint(pos): 
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor


  def calculate_closest_food(self, gameState):
    """
    Calculate the closest food distance
    Returns None if no food is left
    """
    foodList = self.getFood(gameState).asList()
    if foodList:
      return min([self.getMazeDistance(gameState.getAgentPosition(self.index), food) for food in foodList])
    else:
      return None

  def calculate_ghosts_1_step_away(self, gameState, action):
    """
    Calculate the number of ghosts 1 step away
    """
    ghosts = []
    opponents = self.getOpponents(gameState)
    if opponents:
      for opponent in opponents:
        if not gameState.getAgentState(opponent).isPacman:
          ghosts.append(gameState.getAgentPosition(opponent))
          
    x, y = gameState.getAgentPosition(self.index)
    dx, dy = game.Actions.directionToVector(action)
    self.ix, self.iy = int(x + dx), int(y + dy)
    
    self.numGhosts = sum([(self.ix, self.iy) == ghost for ghost in ghosts])
    
    return self.numGhosts
    
  def calculate_eats_food(self, gameState, action):
    """
    Calculate if the action eats food
    """
    x, y = gameState.getAgentPosition(self.index)
    dx, dy = game.Actions.directionToVector(action)
    self.ix, self.iy = int(x + dx), int(y + dy)
    return self.getFood(gameState)[self.ix][self.iy]

  def updateWeights(self, gameState, action):
    """
    Update the weights
    """
    reward = self.getReward(gameState, action)
    difference = (reward + self.discount * self.getMaxQ(gameState))[0] - self.getQValue(gameState, action)
    for feature in self.features:
      self.weights[feature] += self.alpha * difference * self.features[feature]
    # save_weights()

  def save_weights(self):
    """
    Save weights to file
    """
    json.dump(self.weights, open('weights.json', 'w'))
  






class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)

