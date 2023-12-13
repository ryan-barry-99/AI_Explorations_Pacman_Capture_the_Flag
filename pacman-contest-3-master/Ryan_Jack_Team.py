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
import time
import math


#################
# Team creation #
#################

class SharedData:
    def __init__(self):
        self.invaders = []
        self.nextInvaders = []
        self.offensive_invaders = []
        self.next_offensive_invaders = []
        self.defensive_invaders = []
        self.next_defensive_invaders = []
        self.invaderPositions = []
        self.nextInvaderPositions = []
    def getPositions(self):
      self.invaderPositions = [invader.getPosition() for invader in self.invaders if invader.getPosition() is not None]
      self.nextInvaderPositions = [invader.getPosition() for invader in self.nextInvaders if invader.getPosition() is not None]

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
  shared_data = SharedData()

  # Pass the shared data instance to the agents
  agent1 = eval(first)(firstIndex, shared_data)
  agent2 = eval(second)(secondIndex, shared_data)

  return [agent1, agent2]

##########
# Agents #
##########
RED = 0
BLUE = 1

class QLearningCaptureAgent(CaptureAgent): 
  def __init__(self, index):
    CaptureAgent.__init__(self, index)
    self.index = index
    # self.params = json.load(open(self.param_json, 'r'))
    if "defensive" in self.param_json:
      self.weights = {
          "onDefense": -0.0,
          "stop": 0.0,
          "enemy1_num_carried": -0.0,
          "enemy2_num_carried": 0.0,
          "teammate_food_carried": -0.001999999980402624,
          "reverse": -0.0010000000102018366,
          "invaderDistance": -0.01600001000349613,
          "distance_to_start": -0.06999999931399184,
          "num_invaders": -0.0,
          "distance_from_middle": -0.00499999995090656,
          "distance_from_closest_invader": -0.01600001000349613,
          "scared_distance": -0.003000000030939157,
          "distance_to_home": -0.00999999990211312,
          "invader_captured": -0.0010000049902054607,
          "food_protected": -0.001999999980402624,
          "ambush_location_x": -0.020000012504370158,
          "ambush_location_y": -0.012000007502722097,
          "teammate_location_x": -0.01599999984312099,
          "teammate_location_y": -0.000999999990201312,
          "opponent1_location_x": -0.004999999951003064,
          "opponent1_location_y": -0.0019999999804012256,
          "opponent2_location_x": -0.004999999951003064,
          "opponent2_location_y": -0.002999999970601838,
          "successor_score": 0.0,
          "num_ghosts_one_step_away": 0.0,
          "distance_from_closest_ghost": 0.0,
          "is_home": 0.0,
          "distanceToFood": 0.0,
          "food_eaten": 0.0,
          "avoided_ghost": -1.9999999806969264e-13,
          "carrying_food": 0.0,
          "distance_to_capsule": 0.0,
          "game_score": -0.02599999974523411,
          "bias": -0.000999999990201312
      }
    elif "offensive" in self.param_json:
      self.weights = {
          "onDefense": -6.358846321145177e-161,
          "stop": 0.0,
          "enemy1_num_carried": -0.0019999995151552077,
          "enemy2_num_carried": 0.0,
          "teammate_food_carried": 0.0,
          "reverse": -0.0009999997626478725,
          "invaderDistance": -2.098419282942684e-09,
          "distance_to_start": -0.04299998957553696,
          "num_invaders": -6.358846321145177e-161,
          "distance_from_middle": -0.011999997090931245,
          "distance_from_closest_invader": -2.098419282942684e-09,
          "scared_distance": -0.003999999641921322,
          "distance_to_home": -0.03299999200036093,
          "invader_captured": 0.0010999998688726235,
          "food_protected": -0.02999999272732811,
          "ambush_location_x": -5.722961680446216e-10,
          "ambush_location_y": -1.2717692623213815e-10,
          "teammate_location_x": -0.004999998787888018,
          "teammate_location_y": -0.010999997333253641,
          "opponent1_location_x": -0.019999999903794533,
          "opponent1_location_y": -0.007999999961517814,
          "opponent2_location_x": -0.02199999989417399,
          "opponent2_location_y": -0.009999999951897267,
          "successor_score": 0.0,
          "num_ghosts_one_step_away": 0.0,
          "distance_from_closest_ghost": 0.0,
          "is_home": 0.0,
          "distanceToFood": 0.0,
          "food_eaten": 0.0,
          "avoided_ghost": -0.004000000440756134,
          "carrying_food": 0.0,
          "distance_to_capsule": 0.0,
          "game_score": 0.0259999936970177,
          "bias": -0.0009999997575776038
      }
    # self.params["total_reward"].append(0)
    # self.params["num_episodes"] += 1
    # self.epsilon = self.params["epsilon"][-1]  # Exploration rate
    # self.alpha = self.params["alpha"][-1]      # Learning rate
    # self.discount = self.params["discount"][-1] # Discount factor
    self.epsilon = 0
    self.alpha = 0.1
    self.discount = 0.9
    # Initialize variables
    self.total_reward = 0
    self.previous_position = None
    self.maxHomeDistance = 0
    self.defendingFood = 0
    self.visited = set()
    self.max_feature_size = 9999999999
    if self.index % 2 == 0:
        self.color = RED
    else:
        self.color = BLUE
    random.seed(time.monotonic())
    self.foodDistanceScaler = 10
    self.defensiveFoodDistanceScaler = 1
    self.homeDistanceScaler = 10.001
    self.offensiveInvaderScaler = 1000
    self.defensiveInvaderScaler = 1000
    self.capsuleDistanceScaler = 10
    self.teammateScaler = 1
    self.lastPos = None
    self.leaveHomeScaler = 1000
    self.lastNonCorner = None
    self.offense = False
    
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

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2 and self.offense:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.startPosition,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction
    
    # Remove STOP action
    actions.remove(Directions.STOP)

    for a in actions:
      self.update(gameState, a) 

    maxValue = max(self.qValues)
    bestActions = [a for a, v in zip(actions, self.qValues) if v == maxValue]
    # Choose action using epsilon-greedy policy
    if util.flipCoin(self.epsilon):
      # Explore: choose random action
      return random.choice(actions)
    

    # Exploit: choose action with highest Q-value
    position = gameState.getAgentPosition(self.index)
    bestValue = float("-inf")
    bestAction = None
    for action in actions:
       qValue = self.qValues[(position, action)]
       if qValue > bestValue:
          bestAction = action
          bestValue = qValue
    

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2 and self.offense:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.startPosition,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
    return bestAction
  
  
  
 # def getQValue(self, gameState, action): 
  #  features = self.getFeatures(gameState, action)
 #   return sum(features[f] * self.weights[f] for f in features)


  def getQValue(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weighted_sum = sum(features[f] * self.weights[f] for f in features)
    sigmoid_value = 1 / (1 + math.exp(-weighted_sum))
    return sigmoid_value

  def update(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    current_position = gameState.getAgentPosition(self.index)
    reward = self.getReward(gameState, successor)
    if current_position == self.previous_position:
       reward -= 0.1
    self.qValues[(current_position, action)] = (1 - self.alpha) * self.getQValue(gameState, action) + self.alpha * (reward + self.discount * self.getMaxQ(successor)[0])
    self.previous_position = current_position
    # self.updateWeights(gameState, action)


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
  
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()

    foodList = self.getFood(successor).asList()  

    features["game_score"] = self.getScore(gameState)  

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features["distanceToFood"] = minDistance

    # Distance to start
    features["distance_to_start"] = self.getMazeDistance(myPos, self.startPosition)

    # Weight Bias
    features["bias"] = 1.0

     # Successor score based on food availability
    if action == Directions.STOP:
        features["successor_score"] = self.getScore(successor)
    else:
        if len(foodList) > 0:
            features["successor_score"] = self.getScore(successor) + 1
        else:
            features["successor_score"] = self.getScore(successor) - 1
    
    # Number of ghosts one step away
    ghosts = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    ghosts_one_step_away = [g for g in ghosts if g.getPosition() is not None and self.getMazeDistance(myPos, g.getPosition()) == 1]
    features["num_ghosts_one_step_away"] = len(ghosts_one_step_away)

    # Distance from the closest ghost
    distances = [self.getMazeDistance(myPos, g.getPosition()) for g in ghosts if g.getPosition() is not None]
    if len(distances) > 0:
      closest_ghost_distance = min(distances)
      features["distance_from_closest_ghost"] = closest_ghost_distance

    # Whether the agent is home
    if self.color == RED:
      features["is_home"] = 1 if myPos[0] < gameState.data.layout.width / 2 - 1 else 0
    else:
      features["is_home"] = 1 if myPos[0] > gameState.data.layout.width / 2 + 1 else 0



    # Food eaten
    features["food_eaten"] = self.getFood(gameState)[int(myPos[0])][int(myPos[1])]

    # Avoid ghosts
    if gameState.getAgentState(self.index).isPacman:
      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      ghosts = [e for e in enemies if not e.isPacman and e.getPosition() != None]
      if ghosts:
          minDistance = min([self.getMazeDistance(myPos, g.getPosition()) for g in ghosts])
          if minDistance > 1:
              features["avoided_ghost"] = minDistance
          else:
            features["avoided_ghost"] = 0  

    # Carrying food
    features["carrying_food"] = gameState.getAgentState(self.index).numCarrying
    
    # Distance to capsules
    capsules = self.getCapsules(gameState) 

    if len(capsules) > 0: # This should always be True,  but better safe than sorry
        myPos = successor.getAgentState(self.index).getPosition()
        minDistance = min([self.getMazeDistance(myPos, capsule) for capsule in capsules])
        features["distance_to_capsule"] = minDistance

    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    
    features["game_score"] = self.getScore(gameState)  

    # Computes whether we're on defense (1) or offense (0)

    # feature for how much food each enemy has
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    features["enemy1_num_carried"] = enemies[0].numCarrying
    features["enemy2_num_carried"] = enemies[1].numCarrying


    # scaredAgents = [a for a in enemies if not a.isPacman and a.scaredTimer > 0]
    scaredAgents = [successor.getAgentPosition(i) for i in self.getOpponents(successor) if successor.getAgentState(i).scaredTimer > 0 and successor.getAgentPosition(i) != None]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features["num_invaders"] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features["invaderDistance"] = min(dists)
      features["onDefense"] = 1
    else:
      features["onDefense"] = 0

    if len(scaredAgents) > 0:
        dists = [self.getMazeDistance(myPos, a) for a in scaredAgents if a is not None]
        features["scared_distance"] = min(dists)

    if action == Directions.STOP: features["stop"] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features["reverse"] = 1

    # Distance to start
    features["distance_to_start"] = self.getMazeDistance(myPos, self.startPosition)

    # Weight Bias
    features["bias"] = 1.0

    # Distance from the middle
    distanceFromMiddle = abs(myPos[0] - gameState.data.layout.width / 2)
    features["distance_from_middle"] = distanceFromMiddle

    # Distance from the closest invader
    opponents = self.getOpponents(gameState)
    invaders = [i for i in opponents if gameState.getAgentState(i).isPacman]
    invader_pos = []
    if invaders:
        for invader_index in invaders:
            invader_state = gameState.getAgentState(invader_index)
            pos = invader_state.getPosition()
            if pos is not None:
                invader_pos.append(pos)
        if invader_pos:
            closestInvaderDistance = min([self.getMazeDistance(myPos, pos) for pos in invader_pos])
            for pos in invader_pos:
               if self.getMazeDistance(myPos, pos) == closestInvaderDistance:
                  closestInvaderPos = pos
            features["distance_from_closest_invader"] = closestInvaderDistance
            features["ambush_location_x"], features["ambush_location_y"] = closestInvaderPos
    
    
    teammates = self.getTeam(gameState)
    teammatePositions = [gameState.getAgentPosition(i) for i in teammates if i != self.index]
    features["teammate_location_x"], features["teammate_location_y"] = teammatePositions[0]

    opponents = self.getOpponents(gameState)
    opponentPositions = [gameState.getAgentPosition(i) for i in opponents if gameState.getAgentPosition(i) != None and gameState.getAgentPosition(i) != None]
    if len(opponentPositions) == 2:
      features["opponent1_location_x"], features["opponent1_location_y"] = opponentPositions[0]
      features["opponent2_location_x"], features["opponent2_location_y"] = opponentPositions[1]

    features["teammate_food_carried"] = gameState.getAgentState(teammates[0]).numCarrying

    # Distance to home
    if self.color == RED:
      width = int(gameState.data.layout.width/2 - 2)
    else:
      width = int(gameState.data.layout.width/2 + 2)
    minDistanceToHome = 9999
    minHeight = 0
    for height in range(0, gameState.data.layout.height+2):
      try:
        features["distance_to_home"] = self.getMazeDistance(myPos, (width , height))
        if features["distance_to_home"] < minDistanceToHome:
          minHeight = height
      except:
         continue
    


    # Invaders captured
    if self.gotCaptured(gameState):
      features["invader_captured"] += 1

    # Food left
    foodLeft = len(self.getFood(gameState).asList())
    features["food_protected"] = foodLeft

    # Avoid ghosts
    if gameState.getAgentState(self.index).isPacman:
      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      ghosts = [e for e in enemies if not e.isPacman and e.getPosition() != None]
      if ghosts:
          minDistance = min([self.getMazeDistance(myPos, g.getPosition()) for g in ghosts])
          if minDistance > 1:
              features["avoided_ghost"] = minDistance
          else:
            features["avoided_ghost"] = 0  


    for feature in features:
      if features[feature] > self.max_feature_size:
        features[feature] = self.max_feature_size
      elif features[feature] < -self.max_feature_size:
        features[feature] = -self.max_feature_size
    return features
 
  def gotCaptured(self, gameState):
      """
      Check if the agent has captured invaders in the current state
      """
      myState = gameState.getAgentState(self.index)
      myPos = myState.getPosition()

      opponents = self.getOpponents(gameState)
      invaders = [i for i in opponents if gameState.getAgentState(i).isPacman]
      for invader_index in invaders:
          invader_state = gameState.getAgentState(invader_index)
          invader_pos = invader_state.getPosition()
          if invader_pos is not None and self.getMazeDistance(myPos, invader_pos) <= 1:
              return True

      return False
  


  def isInCorner(self, pos, gameState):
      """
      Check if the agent is in a corner or confined space
      """
      walls = gameState.getWalls()
      x, y = int(pos[0]), int(pos[1])


      # Check if the agent is in a corner
      if walls[x][y]:
          return True

      # Check if the agent is in a confined space
      adjacent_walls = 0
      for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
          if walls[x + dx][y + dy]:
              adjacent_walls += 1
      if adjacent_walls >= 3:
          return True

      return False
  
  def removeCorners(self, positions, gameState):
    """
    Remove positions from the list if they are in corners
    """
    non_corner_positions = []
    for pos in positions:
        if not self.isInCorner(pos, gameState):
            non_corner_positions.append(pos)
    return non_corner_positions


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
    self.reward = self.getReward(gameState, nextState)
    # correction = (reward + self.discountRate*self.getValue(nextState)) - self.getQValue(gameState, action)
    difference = (self.reward + self.discount * self.getMaxQ(nextState)[0]) - self.getQValue(gameState, action)
    features = self.getFeatures(gameState, action)
    for feature in features:
      self.weights[feature] += self.alpha * difference * features[feature]
      if self.weights[feature] > self.max_feature_size:
        self.weights[feature] = min(self.max_feature_size, self.weights[feature])
      elif self.weights[feature] < -self.max_feature_size:
        self.weights[feature] = max(-self.max_feature_size, self.weights[feature])
      self.weights[feature] = self.weights[feature] / self.max_feature_size
    # self.save_weights()

  def save_weights(self):
    """
    Save weights to file
    """
    json.dump(self.params, open(self.param_json, 'w'), indent=4)

  
class QLearningOffensiveAgent(QLearningCaptureAgent):
  def __init__(self, index, shared_data):
    self.param_json = 'offensiveParams3.json'
    super().__init__(index)
    self.shared_data = shared_data
    self.offense = True

    
  
  def getReward(self, gameState, nextState):
    """
    Get the reward for the offensive agent
    """
    reward = 0
    
    
    # Get useful info 
    myPos = gameState.getAgentPosition(self.index)
    foodList = self.getFood(gameState).asList()
    capsules = self.getCapsules(gameState)
    dist = self.getMazeDistance(myPos, self.startPosition)
    closestInvaderDist = 9999
    nextPos = nextState.getAgentState(self.index).getPosition()
    nextdist = self.getMazeDistance(nextPos, self.startPosition)

    score = self.getScore(gameState)
    if self.color == RED:
      if score > 0:
        reward += score
      else:
        reward -= score
      width = int(gameState.data.layout.width/2 - 2)
      if myPos[0] < width:
        isHome = True
      else:
        isHome = False
    else:
      if score < 0:
        reward += score
      else:
        reward -= score
      width = int(gameState.data.layout.width/2 + 2)
      if myPos[0] > width:
        isHome = True
      else:
        isHome = False

    enemies = [nextState.getAgentState(i) for i in self.getOpponents(nextState)]
    reward -= enemies[0].numCarrying
    reward -= enemies[1].numCarrying

    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    nextEnemies = [nextState.getAgentState(i) for i in self.getOpponents(nextState)]
    ghosts = [ghost for ghost in enemies if not ghost.isPacman and ghost.getPosition() is not None]
    nextGhosts = [ghost for ghost in nextEnemies if not ghost.isPacman and ghost.getPosition() is not None]
    ghostPositions = [ghost.getPosition() for ghost in ghosts if ghost.getPosition() is not None]
    nextGhostPositions = [ghost.getPosition() for ghost in nextGhosts if ghost.getPosition() is not None]
    minGhostDistance = 999999
    minNextGhostDistance = 999999
    if len(ghostPositions) > 0:
      minGhostDistance = min([self.getMazeDistance(myPos, ghost) for ghost in ghostPositions])
    if len(nextGhostPositions) > 0:
      minNextGhostDistance = min([self.getMazeDistance(nextPos, ghost) for ghost in nextGhostPositions])
    closestInvaderDist = 999999
    closestNextInvaderDist = 999999
    
      
    if self.color == RED:
      invaders = [gameState.getAgentState(i) for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman if gameState.getAgentState(i).getPosition() is not None and gameState.getAgentState(i).getPosition()[0] < width and gameState.getAgentState(i).scaredTimer == 0]
      nextInvaders = [nextState.getAgentState(i) for i in self.getOpponents(nextState) if nextState.getAgentState(i).isPacman if nextState.getAgentState(i).getPosition() is not None and nextState.getAgentState(i).getPosition()[0] < width and nextState.getAgentState(i).scaredTimer == 0]
    else:
      invaders = [gameState.getAgentState(i) for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman if gameState.getAgentState(i).getPosition() is not None and gameState.getAgentState(i).getPosition()[0] > width and gameState.getAgentState(i).scaredTimer == 0]
      nextInvaders = [nextState.getAgentState(i) for i in self.getOpponents(nextState) if nextState.getAgentState(i).isPacman if nextState.getAgentState(i).getPosition() is not None and nextState.getAgentState(i).getPosition()[0] > width and nextState.getAgentState(i).scaredTimer == 0]
    invaderPositions = [invader.getPosition() for invader in invaders if invader.getPosition() is not None]
    nextInvaderPositions = [invader.getPosition() for invader in nextInvaders if invader.getPosition() is not None]
    # if len(invaderPositions) > 0:
    self.shared_data.offensive_invaders = invaders
    self.shared_data.next_offensive_invaders = nextInvaders
    self.shared_data.invaders = self.shared_data.offensive_invaders + self.shared_data.defensive_invaders
    self.shared_data.nextInvaders = self.shared_data.next_offensive_invaders + self.shared_data.next_defensive_invaders
    self.shared_data.getPositions()

    if len(self.shared_data.nextInvaders) > 0:
      closestNextInvaderDist = min([self.getMazeDistance(nextPos, invader) for invader in self.shared_data.nextInvaderPositions])
    if len(self.shared_data.invaders) > 0:
      # print("here")
      closestInvaderDist = min([self.getMazeDistance(myPos, invader) for invader in self.shared_data.invaderPositions])
    


    distsToInvaders = [self.getMazeDistance(myPos, invader) for invader in invaderPositions if self.shared_data.invaderPositions is not None]
    distsToNextInvaders = [self.getMazeDistance(nextPos, invader) for invader in invaderPositions if self.shared_data.invaderPositions is not None]
    # print(self.shared_data.invaders)
    if len(self.shared_data.invaders) == 0 or gameState.getAgentState(self.index).scaredTimer > 0:
      # Reward for getting closer to home
      if self.color == RED:
        width = int(gameState.data.layout.width/2 - 1)
        equator = int(gameState.data.layout.height/2)
        prime_meriderian = int((width)/2)


      else:
        width = int(gameState.data.layout.width/2 + 1)
        equator = int(gameState.data.layout.height/2)
        prime_meriderian = int((width)/2) + width


      foodList = self.getFood(gameState).asList()

      filteredFoodList = []
      quads = [False, False, False, False]
      for pos in ghostPositions:
        if pos[0] <= prime_meriderian:
          if pos[1] < equator:
            quads[0] = True
          else:
            quads[1] = True
        else:
          if pos[1] < equator:
            quads[2] = True
          else:
            quads[3] = True
      for food in foodList:
        if food[0] <= prime_meriderian:
          if food[1] < equator:
            if quads[0]:
              continue
          else:
            if quads[1]:
              continue
        else:
          if food[1] < equator:
            if quads[2]:
              continue
          else:
            if quads[3]:
              continue
        filteredFoodList.append(food)
      
      if len(filteredFoodList) < 0:
        distToFood = min([self.getMazeDistance(myPos, food) for food in filteredFoodList])
        distToNextFood = min([self.getMazeDistance(nextPos, food) for food in filteredFoodList])
      else:
        distToFood = min([self.getMazeDistance(myPos, food) for food in foodList])
        distToNextFood = min([self.getMazeDistance(nextPos, food) for food in foodList])
      # print(closestInvaderDist)
      if distToNextFood < minGhostDistance - 4 or distToNextFood < closestInvaderDist - 4:
        if distToFood > distToNextFood:
          reward += self.foodDistanceScaler / max(distToNextFood, 0.1)
        else:
          reward -= self.foodDistanceScaler / max(distToNextFood, 0.1)
      else:
        if minGhostDistance > minNextGhostDistance - 4 or distToNextFood > closestInvaderDist - 4:
          reward += self.foodDistanceScaler / max(minNextGhostDistance, 0.0001)
        else:
          reward -= self.foodDistanceScaler / max(minNextGhostDistance, 0.0001)
          
      if not self.isInCorner(myPos, gameState):
        self.lastNonCorner = myPos

      if self.lastPos == nextPos:
        reward -= 3
      self.lastPos = myPos
      
      
      minDistanceToHome = 9999
      minHeight = 0
      for height in range(0, gameState.data.layout.height):
        try:
          distanceToHome = self.getMazeDistance(nextPos, (width , height))
          if distanceToHome <= minDistanceToHome:
            minDistanceToHome = distanceToHome
            minHeight = height
        except:
          continue
      if gameState.getAgentState(self.index).numCarrying > 0:
        nextdistanceToHome = self.getMazeDistance(nextPos, (width, minHeight))
        distanceToHome = self.getMazeDistance(myPos, (width, minHeight))
        if nextdistanceToHome < distanceToHome:
          reward += self.homeDistanceScaler * gameState.getAgentState(self.index).numCarrying / max(distanceToHome, 0.0000000000000000001)
        else:
          reward -= self.homeDistanceScaler * gameState.getAgentState(self.index).numCarrying / max(distanceToHome, 0.0000000000000000001)
        if gameState.data.timeleft < 200:
          self.foodDistanceScaler = 0
          self.homeDistanceScaler = 99999999999999

      if dist > 2 and nextPos == self.startPosition:
        # I die in the next state
        reward = -99999999
        self.maxHomeDistance = 0
        self.visited = set()

      
      # if nextdist > self.maxHomeDistance and nextdist < 20 and nextPos not in self.visited:
      #     self.maxHomeDistance = nextdist
      #     reward += self.leaveHomeScaler / max(20 - dist, 0.00001)
      #     self.visited.add(myPos)
      # else:
      #     reward -= self.leaveHomeScaler / max(20 - dist, 0.00001)

        
      if self.color == RED:
        invaders = [gameState.getAgentState(i) for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman if gameState.getAgentState(i).getPosition() is not None and gameState.getAgentState(i).getPosition()[0] < width and gameState.getAgentState(i).scaredTimer == 0]
        nextInvaders = [nextState.getAgentState(i) for i in self.getOpponents(nextState) if nextState.getAgentState(i).isPacman if nextState.getAgentState(i).getPosition() is not None and nextState.getAgentState(i).getPosition()[0] < width and nextState.getAgentState(i).scaredTimer == 0]
      else:
        invaders = [gameState.getAgentState(i) for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman if gameState.getAgentState(i).getPosition() is not None and gameState.getAgentState(i).getPosition()[0] > width and gameState.getAgentState(i).scaredTimer == 0]
        nextInvaders = [nextState.getAgentState(i) for i in self.getOpponents(nextState) if nextState.getAgentState(i).isPacman if nextState.getAgentState(i).getPosition() is not None and nextState.getAgentState(i).getPosition()[0] > width and nextState.getAgentState(i).scaredTimer == 0]
      invaderPositions = [invader.getPosition() for invader in invaders if invader.getPosition() is not None]
      nextInvaderPositions = [invader.getPosition() for invader in nextInvaders if invader.getPosition() is not None]
      # if len(invaderPositions) > 0:
      self.shared_data.defensive_invaders = invaders
      self.shared_data.next_defensive_invaders = nextInvaders
      self.shared_data.invaders = self.shared_data.offensive_invaders + self.shared_data.defensive_invaders
      self.shared_data.nextInvaders = self.shared_data.next_offensive_invaders + self.shared_data.next_defensive_invaders
      self.shared_data.getPositions()

      if len(self.shared_data.nextInvaders) > 0:
        closestNextInvaderDist = min([self.getMazeDistance(nextPos, invader) for invader in self.shared_data.nextInvaderPositions])
      if len(self.shared_data.invaders) > 0:
        closestInvaderDist = min([self.getMazeDistance(myPos, invader) for invader in self.shared_data.invaderPositions])

      closestInvaderDist = 999999
      closestNextInvaderDist = 999999
      
      distsToInvaders = [self.getMazeDistance(myPos, invader) for invader in invaderPositions if invaderPositions is not None]
      distsToNextInvaders = [self.getMazeDistance(nextPos, invader) for invader in invaderPositions if invaderPositions is not None]
      maxCarrying = 0
      # if len(self.shared_data.invaders) > 0:
        
      for i, invader in enumerate(invaders):
        if invader.numCarrying > maxCarrying:
          maxCarrying = invader.numCarrying
          invaderDist = self.getMazeDistance(myPos, self.shared_data.invaderPositions[i])
          nextInvaderDist = self.getMazeDistance(nextPos, self.shared_data.invaderPositions[i])
        invaderDist = min(distsToInvaders)
        nextInvaderDist = min(distsToNextInvaders)
      if maxCarrying > 5:
        if nextState.getAgentState(self.index).scaredTimer == 0:
          if invaderDist > nextInvaderDist:
            reward += self.offensiveInvaderScaler / max(nextInvaderDist, 0.0001)
          else:
            reward -= self.offensiveInvaderScaler / max(nextInvaderDist, 0.0001)
        else:
          if invaderDist > nextInvaderDist + 1:
            reward += self.defensiveInvaderScaler / max(invaderDist, 0.0001)
          else:
            reward -= self.defensiveInvaderScaler / max(invaderDist, 0.0001)

    else:
      for i, invader in enumerate(self.shared_data.invaders):
      # reward -= invader.numCarrying ** 2
        # if invader.numCarrying > maxCarrying:
        maxCarrying = invader.numCarrying
        invaderDist = self.getMazeDistance(myPos, self.shared_data.invaderPositions[i])
        nextInvaderDist = self.getMazeDistance(nextPos, self.shared_data.invaderPositions[i])
        # else:
        #   invaderDist = min(distsToInvaders)
        #   nextInvaderDist = min(distsToNextInvaders)
    if len(self.shared_data.invaders) > 0:
      self.FirstInvader = True
      if nextState.getAgentState(self.index).scaredTimer == 0:
        if invaderDist > nextInvaderDist:
          reward += self.defensiveInvaderScaler / max(nextInvaderDist, 0.0001)
        else:
          reward -= self.defensiveInvaderScaler / max(nextInvaderDist, 0.0001)
      else:
        # print(invaderDist, nextInvaderDist)
        if invaderDist > nextInvaderDist and nextInvaderDist:
          reward -= self.defensiveInvaderScaler / max(invaderDist, 0.0001)
        elif invaderDist < nextInvaderDist or nextInvaderDist:
          reward += self.defensiveInvaderScaler / max(invaderDist, 0.0001)

      
    # Distance to capsules
    capsules = self.getCapsules(gameState) 

    if len(capsules) > 0: # This should always be True,  but better safe than sorry
        myPos = nextState.getAgentState(self.index).getPosition()
        minDistanceToCapsule = min([self.getMazeDistance(nextPos, capsule) for capsule in capsules])
        if minGhostDistance < 10:
          reward += self.capsuleDistanceScaler / max(minDistanceToCapsule, 0.0001)

      
    # for catching invaders 
    if self.gotCaptured(gameState):
        reward += 99999999

    # self.params["total_reward"][-1] += reward
    # self.params["latest_reward"] = self.params["total_reward"][-1]
    # self.save_weights()
    return reward

  
  
  
class QLearningDefensiveAgent(QLearningCaptureAgent):
  def __init__(self, index, shared_data):
    self.param_json = 'defensiveParams6.json'
    super().__init__(index)
    self.shared_data = shared_data
    self.i=0
    self.FirstInvader=False
  
      
  def offensiveReward(self, gameState, nextState):
    reward = 0
    myPos = gameState.getAgentPosition(self.index)
    foodList = self.getFood(gameState).asList()
    capsules = self.getCapsules(gameState)
    dist = self.getMazeDistance(myPos, self.startPosition)
    closestInvaderDist = 9999
    nextPos = nextState.getAgentState(self.index).getPosition()
    nextdist = self.getMazeDistance(nextPos, self.startPosition)

    score = self.getScore(gameState)
    if self.color == RED:
      if score > 0:
        reward += score
      else:
        reward -= score
      width = int(gameState.data.layout.width/2 - 2)
      if myPos[0] < width:
        isHome = True
      else:
        isHome = False
    else:
      if score < 0:
        reward += score
      else:
        reward -= score
      width = int(gameState.data.layout.width/2 + 2)
      if myPos[0] > width:
        isHome = True
      else:
        isHome = False

    enemies = [nextState.getAgentState(i) for i in self.getOpponents(nextState)]
    reward -= enemies[0].numCarrying
    reward -= enemies[1].numCarrying

    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    nextEnemies = [nextState.getAgentState(i) for i in self.getOpponents(nextState)]
    ghosts = [ghost for ghost in enemies if not ghost.isPacman and ghost.getPosition() is not None]
    nextGhosts = [ghost for ghost in nextEnemies if not ghost.isPacman and ghost.getPosition() is not None]
    ghostPositions = [ghost.getPosition() for ghost in ghosts if ghost.getPosition() is not None]
    nextGhostPositions = [ghost.getPosition() for ghost in nextGhosts if ghost.getPosition() is not None]
    minGhostDistance = 999999
    minNextGhostDistance = 999999
    if len(ghostPositions) > 0:
      minGhostDistance = min([self.getMazeDistance(myPos, ghost) for ghost in ghostPositions])
    if len(nextGhostPositions) > 0:
      minNextGhostDistance = min([self.getMazeDistance(nextPos, ghost) for ghost in nextGhostPositions])
    closestInvaderDist = 999999
    closestNextInvaderDist = 999999
    
      
    if self.color == RED:
      invaders = [gameState.getAgentState(i) for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman if gameState.getAgentState(i).getPosition() is not None and gameState.getAgentState(i).getPosition()[0] < width and gameState.getAgentState(i).scaredTimer == 0]
      nextInvaders = [nextState.getAgentState(i) for i in self.getOpponents(nextState) if nextState.getAgentState(i).isPacman if nextState.getAgentState(i).getPosition() is not None and nextState.getAgentState(i).getPosition()[0] < width and nextState.getAgentState(i).scaredTimer == 0]
    else:
      invaders = [gameState.getAgentState(i) for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman if gameState.getAgentState(i).getPosition() is not None and gameState.getAgentState(i).getPosition()[0] > width and gameState.getAgentState(i).scaredTimer == 0]
      nextInvaders = [nextState.getAgentState(i) for i in self.getOpponents(nextState) if nextState.getAgentState(i).isPacman if nextState.getAgentState(i).getPosition() is not None and nextState.getAgentState(i).getPosition()[0] > width and nextState.getAgentState(i).scaredTimer == 0]
    invaderPositions = [invader.getPosition() for invader in invaders if invader.getPosition() is not None]
    nextInvaderPositions = [invader.getPosition() for invader in nextInvaders if invader.getPosition() is not None]
    # if len(invaderPositions) > 0:
    self.shared_data.offensive_invaders = invaders
    self.shared_data.next_offensive_invaders = nextInvaders
    self.shared_data.invaders = self.shared_data.offensive_invaders + self.shared_data.defensive_invaders
    self.shared_data.nextInvaders = self.shared_data.next_offensive_invaders + self.shared_data.next_defensive_invaders
    self.shared_data.getPositions()

    if len(self.shared_data.nextInvaders) > 0:
      closestNextInvaderDist = min([self.getMazeDistance(nextPos, invader) for invader in self.shared_data.nextInvaderPositions])
    if len(self.shared_data.invaders) > 0:
      # print("here")
      closestInvaderDist = min([self.getMazeDistance(myPos, invader) for invader in self.shared_data.invaderPositions])
    


    distsToInvaders = [self.getMazeDistance(myPos, invader) for invader in invaderPositions if self.shared_data.invaderPositions is not None]
    distsToNextInvaders = [self.getMazeDistance(nextPos, invader) for invader in invaderPositions if self.shared_data.invaderPositions is not None]
    # print(self.shared_data.invaders)
    if len(self.shared_data.invaders) == 0 or gameState.getAgentState(self.index).scaredTimer > 0:
      # Reward for getting closer to home
      if self.color == RED:
        width = int(gameState.data.layout.width/2 - 1)
        equator = int(gameState.data.layout.height/2)
        prime_meriderian = int((width)/2)


      else:
        width = int(gameState.data.layout.width/2 + 1)
        equator = int(gameState.data.layout.height/2)
        prime_meriderian = int((width)/2) + width


      foodList = self.getFood(gameState).asList()

      filteredFoodList = []
      quads = [False, False, False, False]
      for pos in ghostPositions:
        if pos[0] <= prime_meriderian:
          if pos[1] < equator:
            quads[0] = True
          else:
            quads[1] = True
        else:
          if pos[1] < equator:
            quads[2] = True
          else:
            quads[3] = True
      for food in foodList:
        if food[0] <= prime_meriderian:
          if food[1] < equator:
            if quads[0]:
              continue
          else:
            if quads[1]:
              continue
        else:
          if food[1] < equator:
            if quads[2]:
              continue
          else:
            if quads[3]:
              continue
        filteredFoodList.append(food)
      
      if len(filteredFoodList) < 0:
        distToFood = min([self.getMazeDistance(myPos, food) for food in filteredFoodList])
        distToNextFood = min([self.getMazeDistance(nextPos, food) for food in filteredFoodList])
      else:
        distToFood = min([self.getMazeDistance(myPos, food) for food in foodList])
        distToNextFood = min([self.getMazeDistance(nextPos, food) for food in foodList])
      # print(closestInvaderDist)
      if distToNextFood < minGhostDistance - 4 or distToNextFood < closestInvaderDist - 4:
        if distToFood > distToNextFood:
          reward += self.foodDistanceScaler / max(distToNextFood, 0.1)
        else:
          reward -= self.foodDistanceScaler / max(distToNextFood, 0.1)
      else:
        if minGhostDistance > minNextGhostDistance - 4 or distToNextFood > closestInvaderDist - 4:
          reward += self.foodDistanceScaler / max(minNextGhostDistance, 0.0001)
        else:
          reward -= self.foodDistanceScaler / max(minNextGhostDistance, 0.0001)
          
      if not self.isInCorner(myPos, gameState):
        self.lastNonCorner = myPos

      if self.lastPos == nextPos:
        reward -= 3
      self.lastPos = myPos
      
      
      minDistanceToHome = 9999
      minHeight = 0
      for height in range(0, gameState.data.layout.height):
        try:
          distanceToHome = self.getMazeDistance(nextPos, (width , height))
          if distanceToHome <= minDistanceToHome:
            minDistanceToHome = distanceToHome
            minHeight = height
        except:
          continue
      if gameState.getAgentState(self.index).numCarrying > 0:
        nextdistanceToHome = self.getMazeDistance(nextPos, (width, minHeight))
        distanceToHome = self.getMazeDistance(myPos, (width, minHeight))
        if nextdistanceToHome < distanceToHome:
          reward += self.homeDistanceScaler * gameState.getAgentState(self.index).numCarrying / max(distanceToHome, 0.0000000000000000001)
        else:
          reward -= self.homeDistanceScaler * gameState.getAgentState(self.index).numCarrying / max(distanceToHome, 0.0000000000000000001)
        
      if dist > 2 and nextPos == self.startPosition:
        # I die in the next state
        reward = -99999999
        self.maxHomeDistance = 0
        self.visited = set()

      
      # if nextdist > self.maxHomeDistance and nextdist < 20 and nextPos not in self.visited:
      #     self.maxHomeDistance = nextdist
      #     reward += self.leaveHomeScaler / max(20 - dist, 0.00001)
      #     self.visited.add(myPos)
      # else:
      #     reward -= self.leaveHomeScaler / max(20 - dist, 0.00001)

        
      if self.color == RED:
        invaders = [gameState.getAgentState(i) for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman if gameState.getAgentState(i).getPosition() is not None and gameState.getAgentState(i).getPosition()[0] < width and gameState.getAgentState(i).scaredTimer == 0]
        nextInvaders = [nextState.getAgentState(i) for i in self.getOpponents(nextState) if nextState.getAgentState(i).isPacman if nextState.getAgentState(i).getPosition() is not None and nextState.getAgentState(i).getPosition()[0] < width and nextState.getAgentState(i).scaredTimer == 0]
      else:
        invaders = [gameState.getAgentState(i) for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman if gameState.getAgentState(i).getPosition() is not None and gameState.getAgentState(i).getPosition()[0] > width and gameState.getAgentState(i).scaredTimer == 0]
        nextInvaders = [nextState.getAgentState(i) for i in self.getOpponents(nextState) if nextState.getAgentState(i).isPacman if nextState.getAgentState(i).getPosition() is not None and nextState.getAgentState(i).getPosition()[0] > width and nextState.getAgentState(i).scaredTimer == 0]
      invaderPositions = [invader.getPosition() for invader in invaders if invader.getPosition() is not None]
      nextInvaderPositions = [invader.getPosition() for invader in nextInvaders if invader.getPosition() is not None]
      # if len(invaderPositions) > 0:
      self.shared_data.defensive_invaders = invaders
      self.shared_data.next_defensive_invaders = nextInvaders
      self.shared_data.invaders = self.shared_data.offensive_invaders + self.shared_data.defensive_invaders
      self.shared_data.nextInvaders = self.shared_data.next_offensive_invaders + self.shared_data.next_defensive_invaders
      self.shared_data.getPositions()

      if len(self.shared_data.nextInvaders) > 0:
        closestNextInvaderDist = min([self.getMazeDistance(nextPos, invader) for invader in self.shared_data.nextInvaderPositions])
      if len(self.shared_data.invaders) > 0:
        closestInvaderDist = min([self.getMazeDistance(myPos, invader) for invader in self.shared_data.invaderPositions])

      closestInvaderDist = 999999
      closestNextInvaderDist = 999999
      
      distsToInvaders = [self.getMazeDistance(myPos, invader) for invader in invaderPositions if invaderPositions is not None]
      distsToNextInvaders = [self.getMazeDistance(nextPos, invader) for invader in invaderPositions if invaderPositions is not None]
      maxCarrying = 0
      # if len(self.shared_data.invaders) > 0:
        
      for i, invader in enumerate(invaders):
        if invader.numCarrying > maxCarrying:
          maxCarrying = invader.numCarrying
          invaderDist = self.getMazeDistance(myPos, self.shared_data.invaderPositions[i])
          nextInvaderDist = self.getMazeDistance(nextPos, self.shared_data.invaderPositions[i])
        invaderDist = min(distsToInvaders)
        nextInvaderDist = min(distsToNextInvaders)
      if maxCarrying > 5:
        if nextState.getAgentState(self.index).scaredTimer == 0:
          if invaderDist > nextInvaderDist:
            reward += self.offensiveInvaderScaler / max(nextInvaderDist, 0.0001)
          else:
            reward -= self.offensiveInvaderScaler / max(nextInvaderDist, 0.0001)
        else:
          if invaderDist > nextInvaderDist + 1:
            reward += self.defensiveInvaderScaler / max(invaderDist, 0.0001)
          else:
            reward -= self.defensiveInvaderScaler / max(invaderDist, 0.0001)
    return reward
  

  def getReward(self, gameState, nextState):
    FirstInvader=False
    """
    Get the reward for the defensive agent
    """
    reward = 0
    # if gameState.getAgentState(self.index).scaredTimer > 0:
    #   return self.offensiveReward(gameState, nextState)

    myPos = gameState.getAgentPosition(self.index)
    dist = self.getMazeDistance(myPos, self.startPosition)
    nextPos = nextState.getAgentState(self.index).getPosition()
    nextdist = self.getMazeDistance(nextPos, self.startPosition)

    foodList = self.getFoodYouAreDefending(gameState).asList()
    nextFoodList = self.getFoodYouAreDefending(nextState).asList()
    

    # if dist > 2 and nextPos == self.startPosition:
    #   # I die in the next state
    #   reward = -99999999
    #   self.maxHomeDistance = 0
    #   self.visited = set()

    # if nextdist > self.maxHomeDistance and nextdist < 20 and nextPos not in self.visited:
    #     self.maxHomeDistance = nextdist
    #     reward += self.leaveHomeScaler / max(20 - dist, 0.00001)
    #     self.visited.add(myPos)
    # else:
    #     reward -= self.leaveHomeScaler / max(20 - dist, 0.00001)
    # Reward for getting closer to home
    if self.color == RED:
      width = int(gameState.data.layout.width/2 - 2)
      if myPos[0] < width:
        isHome = True
      else:
        isHome = False
    else:
      width = int(gameState.data.layout.width/2 + 2)
      if myPos[0] > width:
        isHome = True
      else:
        isHome = False
    minDistanceToHome = 9999
    minHeight = 0
    for height in range(0, gameState.data.layout.height):
      try:
        distanceToHome = self.getMazeDistance(nextPos, (width , height))
        if distanceToHome <= minDistanceToHome:
          minDistanceToHome = distanceToHome
          minHeight = height
      except:
        continue
    nextdistanceToHome = self.getMazeDistance(nextPos, (width, minHeight))
    distanceToHome = self.getMazeDistance(myPos, (width, minHeight))

    if self.color == RED:
      invaders = [gameState.getAgentState(i) for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman if gameState.getAgentState(i).getPosition() is not None and gameState.getAgentState(i).getPosition()[0] < width and gameState.getAgentState(i).scaredTimer == 0]
    else:
      invaders = [gameState.getAgentState(i) for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman if gameState.getAgentState(i).getPosition() is not None and gameState.getAgentState(i).getPosition()[0] > width and gameState.getAgentState(i).scaredTimer == 0]
    invaderPositions = [invader.getPosition() for invader in invaders if invader.getPosition() is not None]
    
    # if len(invaderPositions) > 0:
    self.shared_data.defensive_invaders = invaders
    self.shared_data.invaders = self.shared_data.offensive_invaders + self.shared_data.defensive_invaders
    self.shared_data.getPositions()
    
    self.foodDistanceScaler = 0
    for i, invader in enumerate(self.shared_data.invaders):
        invaderDist = self.getMazeDistance(myPos, self.shared_data.invaderPositions[i])
        nextInvaderDist = self.getMazeDistance(nextPos, self.shared_data.invaderPositions[i])
    if len(self.shared_data.invaders) > 0:
      self.FirstInvader = True
      if nextState.getAgentState(self.index).scaredTimer == 0:
        if invaderDist > nextInvaderDist:
          reward += self.defensiveInvaderScaler / max(nextInvaderDist, 0.0001)
        else:
          # if self.getFoodYouAreDefending(gameState).count() > 5:
          reward -= self.defensiveInvaderScaler / max(nextInvaderDist, 0.0001)
          # else:
          #   reward += self.defensiveInvaderScaler / max(nextInvaderDist, 0.0001)
      else:
        # print(invaderDist, nextInvaderDist)
        if invaderDist > nextInvaderDist:
          reward += self.defensiveInvaderScaler / max(invaderDist, 0.0001)
        elif invaderDist < nextInvaderDist:
          reward -= self.defensiveInvaderScaler / max(invaderDist, 0.0001)
      
      foodList = self.getFoodYouAreDefending(gameState).asList()
      self.sortedFood = sorted(foodList, key=lambda x: self.getMazeDistance(myPos, x))
      self.i = 0

    else:
      foodList = self.getFoodYouAreDefending(gameState).asList()
      x = self.startPosition[0]

      # Create a new list to store the filtered foods
      filteredFoodList = []

      # Iterate through the foodList
      for food in foodList:
          if food[0] != x:
              # Add the food to the filteredFoodList if the condition is not met
              filteredFoodList.append(food)

      # Update the foodList with the filteredFoodList
      foodList = filteredFoodList


      if self.i == len(foodList):
        self.i = 0
      # Sort the food positions based on their distance to the agent
      if self.i == 0:
        try:
          self.sortedFood = sorted(foodList, key=lambda x: self.getMazeDistance(myPos, x))
        except:
          self.i += 1
          return reward
      try:
        if myPos == self.sortedFood[self.i]:
          self.i += 1
      except:
        self.i = 0
      try:
        distToFood = self.getMazeDistance(myPos, self.sortedFood[self.i])
        nextDistToFood = self.getMazeDistance(nextPos, self.sortedFood[self.i])
        if nextDistToFood < distToFood:
            reward += 1  # Encourage moving closer to the food
        elif nextDistToFood > distToFood:
            reward -= 1  # Discourage moving away from the food
      except:
        return reward


    # self.params["total_reward"][-1] += reward
    # self.params["latest_reward"] = self.params["total_reward"][-1]
    # self.save_weights()
    return reward
  
