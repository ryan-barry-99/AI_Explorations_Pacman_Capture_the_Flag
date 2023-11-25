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
               first = 'QLearningOffensiveAgent', second = 'QLearningDefensiveAgent'):
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
  def __init__(self, index):
    CaptureAgent.__init__(self, index)
    self.index = index
    # Initialize variables
    self.total_reward = 0
    self.epsilon = 0.7  # Exploration rate
    self.alpha = 0.2      # Learning rate
    self.discount = 0.8 # Discount factor
    
  def registerInitialState(self, gameState):
    """
    Initialize agent's state at the start of the game or a new round
    """
    # Call the parent class's registerInitialState() method
    super().registerInitialState(gameState)
    # Initialize Q-values dictionary
    self.qValues = util.Counter()
    self.startPosition = gameState.getAgentPosition(self.index)
  
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
      self.update(gameState, best_action)
      self.updateWeights(gameState, best_action)
      return best_action
  
  
  
  def getQValue(self, gameState, action): 
    return sum(self.getFeatures(gameState, action)[f] * self.weights[f] for f in self.getFeatures(gameState, action))


  def update(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    reward = self.getReward(gameState)
    self.qValues[(gameState, action)] = (1 - self.alpha) * self.getQValue(gameState, action) + self.alpha * (reward + self.discount * self.getMaxQ(successor)[0])


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
      best_action = Directions.STOP
      for action in legalActions:
        qval = self.getQValue(gameState, action)
        qvals.append(qval)
        if qval > maxq:
          maxq = qval
          best_action = action        
    return maxq, best_action
 


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

  def updateWeights(self, gameState, action):
    """
    Update the weights
    """
    nextState = self.getSuccessor(gameState, action)
    self.reward = self.getReward(gameState)
    # correction = (reward + self.discountRate*self.getValue(nextState)) - self.getQValue(gameState, action)
    difference = (self.reward + self.discount * self.getMaxQ(nextState)[0]) - self.getQValue(gameState, action)
    features = self.getFeatures(gameState, action)
    for feature in features:
      self.weights[feature] += self.alpha * difference * features[feature]
    # print(self.weights)
    self.save_weights()

  
class QLearningOffensiveAgent(QLearningCaptureAgent):
  def __init__(self, index):
    super().__init__(index)
    self.params = json.load(open('offensiveParams.json', 'r'))
    self.weights = self.params['weights']
    self.params["total_reward"].append(0)
    self.epsilon = self.params['epsilon'][-1]  # Exploration rate
    self.alpha = self.params['alpha'][-1]      # Learning rate
    self.discount = self.params['discount'][-1] # Discount factor
    
  
  def getFeatures(self, gameState, action):
    features = util.Counter()

    # Distance to closest food
    myPos = gameState.getAgentState(self.index).getPosition()
    foodList = self.getFood(gameState).asList()  
    if foodList:
        minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
        features['distance_to_food'] = 1.0 / minDistance

    # Food eaten
    features['food_eaten'] = self.getFood(gameState)[int(myPos[0])][int(myPos[1])]

    # Avoid ghosts
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghosts = [e for e in enemies if not e.isPacman and e.getPosition() != None]
    if ghosts:
        minDistance = min([self.getMazeDistance(myPos, g.getPosition()) for g in ghosts])
        if minDistance > 1:
            features['avoided_ghost'] = 1 * 0.7

    # Carrying food
    features['carrying_food'] = gameState.getAgentState(self.index).numCarrying
    
    # Distance to capsules
    capsules = self.getCapsulesYouAreDefending(gameState) 
    if capsules:
        minDistance = min([self.getMazeDistance(myPos, c) for c in capsules])
        if minDistance > 1:
            features['distance_to_capsule'] = 1.0 / minDistance

    return features
  
  def getReward(self, gameState):
    """
    Get the reward for the state 
    """
    reward = 0

    # Get useful info 
    myPos = gameState.getAgentPosition(self.index)
    foodList = self.getFood(gameState).asList()
    capsules = self.getCapsules(gameState)
    
    # Reward for eating food
    if gameState.hasFood(myPos[0], myPos[1]):
        reward += 10
        
    # Reward for eating capsule
    if myPos in capsules:
        reward += 50
        
    # Reward for getting closer to food 
    myDist = [self.getMazeDistance(myPos, food) for food in foodList]
    if myDist:
        reward += 1.0/min(myDist)
        
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    enemyGhosts = [a for a in enemies if a.isPacman == False and a.getPosition() != None]
    
    if len(enemyGhosts) > 0:
      myPos = gameState.getAgentState(self.index).getPosition() 
      closestGhostDist = min([self.getMazeDistance(myPos, g.getPosition()) for g in enemyGhosts])  
      if closestGhostDist <= 5:
        capsules = self.getCapsulesYouCanEat(gameState)  
        if len(capsules) > 0:
          reward += 10 # Additional reward for going after capsule when ghost is close
    self.params["total_reward"][-1] += reward
    return reward
  
  def save_weights(self):
    """
    Save weights to file
    """
    json.dump(self.params, open('offensiveParams.json', 'w'))
  
class QLearningDefensiveAgent(QLearningCaptureAgent):
  def __init__(self, index):
    super().__init__(index)
    self.params = json.load(open('defensiveParams.json', 'r'))
    self.weights = self.params['weights']
    self.params["total_reward"].append(0)
    self.epsilon = self.params['epsilon'][-1]  # Exploration rate
    self.alpha = self.params['alpha'][-1]      # Learning rate
    self.discount = self.params['discount'][-1] # Discount factor
  
  def getFeatures(self, gameState, action):
    features = util.Counter()

    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()

    # Distance to home
    dist = self.getMazeDistance(myPos, self.startPosition)
    maxDist = util.manhattanDistance((0,0), (gameState.data.layout.width - 1, gameState.data.layout.height - 1))
    if dist:
        features['distance_to_home'] = self.weights['distance_to_home'] * (1.0 - dist / maxDist)
    

    # Invaders captured
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['invader_captured'] = self.weights['invader_captured'] * len(invaders)

    # Food left
    foodLeft = len(self.getFood(gameState).asList())
    features['food_protected'] = self.weights['food_protected'] * foodLeft

    # Avoided ghosts
    if features['invader_captured'] > 0:
        features['avoided_ghost'] = self.weights['avoided_ghost']

    # Ambush spots (would need more logic here)
    features['ambush_location'] = self.weights['ambush_location']

    return features
  
  def getInvaders(self, gameState):
        """
        Get the invaders in the current state
        """
        invaders = []
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]

        for enemy in enemies:
            if enemy.isPacman and enemy.getPosition() is not None:
                invaders.append(enemy.getPosition())

        return invaders
      
  def gotCaptured(self, gameState):
        """
        Check if the agent has captured invaders in the current state
        """
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()

        opponents = self.getOpponents(gameState)
        invaders = [gameState.getAgentState(i) for i in opponents if gameState.getAgentState(i).isPacman]

        for invader in invaders:
            invaderPos = invader.getPosition()
            if invaderPos is not None and self.getMazeDistance(myPos, invaderPos) <= 1:
                return True

        return False
      
  def getDistanceToHome(self, gameState):
        """
        Calculate the distance from the current position to the friendly side's center
        """
        myPos = gameState.getAgentPosition(self.index)
        home = self.startPosition

        # Adjust home position based on team (red or blue)
        if self.red:
            home = (gameState.data.layout.width // 2 - 1, gameState.data.layout.height // 2 - 1)
        else:
            home = (gameState.data.layout.width // 2, gameState.data.layout.height // 2)

        return self.getMazeDistance(myPos, home)
      
  def getReward(self, gameState):
    reward = 0

    # Reward for each food dot left in our side
    for food in self.getFoodYouAreDefending(gameState).asList():
        reward += 1
    
    # Penalize if invaders are in our side
    for invader in self.getInvaders(gameState): 
        reward -= 2

    # Big reward for catching invaders 
    if self.gotCaptured(gameState):
        reward += 10

    # Reward for staying closer to our side
    dist = self.getDistanceToHome(gameState) 
    if dist <= 5:
        reward += 0.5
    self.params["total_reward"][-1] += reward
    return reward
  
  def save_weights(self):
    """
    Save weights to file
    """
    json.dump(self.params, open('defensiveParams.json', 'w'))
  
     






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

