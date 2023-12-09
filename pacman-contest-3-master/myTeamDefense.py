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

def createTeam(firstIndex, secondIndex, isRed,
               first = 'QLearningDefensiveAgent', second = 'QLearningDefensiveAgent'):
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
RED = 0
BLUE = 1

class QLearningCaptureAgent(CaptureAgent): 
  def __init__(self, index):
    CaptureAgent.__init__(self, index)
    self.index = index
    self.params = json.load(open(self.param_json, 'r'))
    self.weights = self.params["weights"]
    self.params["total_reward"].append(0)
    self.params["num_episodes"] += 1
    self.epsilon = self.params["epsilon"][-1]  # Exploration rate
    self.alpha = self.params["alpha"][-1]      # Learning rate
    self.discount = self.params["discount"][-1] # Discount factor
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
    self.homeDistanceScaler = 10.00001
    self.offensiveInvaderScaler = 100
    self.defensiveInvaderScaler = 100
    self.capsuleDistanceScaler = 11
    self.teammateScaler = 1
    self.lastPos = None
    self.leaveHomeScaler = 1000
    self.lastNonCorner = None
    
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

    if foodLeft <= 2:
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

    if foodLeft <= 2:
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
    self.updateWeights(gameState, action)


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
    self.save_weights()

  def save_weights(self):
    """
    Save weights to file
    """
    json.dump(self.params, open(self.param_json, 'w'), indent=4)

  
class QLearningOffensiveAgent(QLearningCaptureAgent):
  def __init__(self, index):
    self.param_json = 'offensiveParams3.json'
    super().__init__(index)
    

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

    enemies = [nextState.getAgentState(i) for i in self.getOpponents(nextState)]
    reward -= enemies[0].numCarrying
    reward -= enemies[1].numCarrying

    minFoodDistance = 9999
    for food in foodList:
      if self.getMazeDistance(myPos, food) < minFoodDistance:
        minFoodDistance = self.getMazeDistance(myPos, food)
        
    if not self.isInCorner(myPos, gameState):
      self.lastNonCorner = myPos

    if self.lastPos == nextPos:
      reward -= 3
    self.lastPos = myPos
    
    score = self.getScore(gameState)
    if self.color == RED:
      if score > 0:
        reward += score
      else:
        reward -= score
      width = int(gameState.data.layout.width/2 - 1)
      if myPos[0] < width:
        isHome = True
      else:
        isHome = False
    else:
      if score < 0:
        reward += score
      else:
        reward -= score
      width = int(gameState.data.layout.width/2 + 1)
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

    
    if nextdist > self.maxHomeDistance and nextdist < 20 and nextPos not in self.visited:
        self.maxHomeDistance = nextdist
        reward += self.leaveHomeScaler / max(20 - dist, 0.00001)
        self.visited.add(myPos)
    else:
        reward -= self.leaveHomeScaler / max(20 - dist, 0.00001)

      
    invaders = [gameState.getAgentState(i) for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman and gameState.getAgentState(i).getPosition() is not None]
    invaderPositions = [invader.getPosition() for invader in invaders if invader.getPosition() is not None]

    closestInvaderDist = 999999
    closestNextInvaderDist = 999999
    
    distsToInvaders = [self.getMazeDistance(myPos, invader) for invader in invaderPositions if invaderPositions is not None]
    distsToNextInvaders = [self.getMazeDistance(nextPos, invader) for invader in invaderPositions if invaderPositions is not None]
    maxCarrying = 0
    if len(distsToInvaders) > 0 and len(distsToNextInvaders) > 0:
      
      for i, invader in enumerate(invaders):
        if invader.numCarrying > maxCarrying:
          maxCarrying = invader.numCarrying
          invaderDist = self.getMazeDistance(myPos, invaderPositions[i])
          nextInvaderDist = self.getMazeDistance(nextPos, invaderPositions[i])
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
      # Add negative reward for being near teammate
      teammates = self.getTeam(gameState)
      teammatePositions = [gameState.getAgentPosition(i) for i in teammates if i != self.index]
      minTeammateDist = min([self.getMazeDistance(myPos, pos) for pos in teammatePositions])
      nextMinTeammateDist = min([self.getMazeDistance(nextPos, pos) for pos in teammatePositions])
      if not isHome:
        if minTeammateDist > nextMinTeammateDist:
          reward -= self.teammateScaler / max(nextMinTeammateDist, 0.0001)
        elif minTeammateDist < nextMinTeammateDist:
          reward += self.teammateScaler / max(nextMinTeammateDist, 0.0001)  # Scaled by closeness to teammate
      else:
        if minTeammateDist > nextMinTeammateDist:
          reward += self.teammateScaler / max(nextMinTeammateDist, 0.0001)
        elif minTeammateDist < nextMinTeammateDist:
          reward -= self.teammateScaler / max(nextMinTeammateDist, 0.0001)  # Scaled by closeness to teammate
      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      ghosts = [ghost for ghost in enemies if not ghost.isPacman and ghost.getPosition() is not None]
      if len(ghosts) > 0:
        closestGhostIndex = 0
        minNextGhostDist = 999
        minGhostDist = 999
        minGhostDistToNonCorner = 999
        for i, ghost in enumerate(ghosts):
          if self.getMazeDistance(myPos, ghost.getPosition()) < minGhostDist:
            minGhostDist = self.getMazeDistance(myPos, ghost.getPosition())
            closestGhostIndex = i
        minNextGhostDist = min([self.getMazeDistance(nextPos, ghost.getPosition()) for ghost in ghosts])
        closestGhostToTeammate = min([self.getMazeDistance(teammatePositions[0], ghost.getPosition()) for ghost in ghosts for teammate in teammatePositions])
        minGhostDistToNonCorner = min([self.getMazeDistance(self.lastNonCorner, ghost.getPosition()) for ghost in ghosts])
        if closestGhostToTeammate < 3:
          self.capsuleDistanceScaler = 100
        else:
          self.capsuleDistanceScaler = 11
      # check if food eaten in nextState
      # if gameState.getAgentState(self.index).numCarrying == 0:
      distToFood = min([self.getMazeDistance(myPos, food) for food in foodList])
      distToNextFood = min([self.getMazeDistance(nextPos, food) for food in foodList])
      

      
      invaders = [gameState.getAgentState(i).getPosition() for i in self.getOpponents(gameState) if not gameState.getAgentState(i).isPacman]
    
      closestInvaderDist = 999999
      closestNextInvaderDist = 999999
      
      distsToInvaders = [self.getMazeDistance(myPos, invader) for invader in invaders if invader is not None]
      distsToNextInvaders = [self.getMazeDistance(nextPos, invader) for invader in invaders if invader is not None]
      if len(distsToInvaders) > 0 and len(distsToNextInvaders) > 0:
        closestInvaderDist = min(distsToInvaders)
        closestNextInvaderDist = min(distsToNextInvaders)
      
      

      
      if distToNextFood < closestNextInvaderDist or (isHome and minDistanceToHome < 3):
        if distToFood > distToNextFood:
          reward += self.foodDistanceScaler / max(distToNextFood, 0.1)
        else:
          reward -= self.foodDistanceScaler / max(distToNextFood, 0.1)
      # else:
      #   if closestInvaderDist > closestNextInvaderDist:
      #     reward -= self.offensiveInvaderScaler / max(closestNextInvaderDist, 0.0001)
      #   else:
      #     reward += self.offensiveInvaderScaler / max(closestNextInvaderDist, 0.0001)

      #   if ghosts[closestGhostIndex].scaredTimer == 0:
      #     if minGhostDist > minNextGhostDist:
      #       reward -= self.offensiveInvaderScaler / max(minNextGhostDist, 0.0001)
      #     elif minGhostDist < minNextGhostDist:
      #       reward += self.offensiveInvaderScaler / max(minNextGhostDist, 0.0001)
      #   else:
      #     if minGhostDist > minNextGhostDist:
      #       reward += self.offensiveInvaderScaler / max(minNextGhostDist, 0.0001)
      #     elif minGhostDist < minNextGhostDist:
      #       reward -= self.offensiveInvaderScaler / max(minNextGhostDist, 0.0001)
        
        # if self.isInCorner(nextPos, gameState):
        #   if minGhostDist > minNextGhostDist and minNextGhostDist < 8 and not isHome:
        #     reward -= 10 / max(minNextGhostDist, 0.0001)
        #   elif minGhostDist < minNextGhostDist and minNextGhostDist < 8 and not isHome:
        #     reward += 10 / max(minNextGhostDist, 0.0001)

      # if gameState.getAgentState(self.index).numCarrying > 0:
      #   nextdistanceToHome = self.getMazeDistance(nextPos, (width, minHeight))
      #   distanceToHome = self.getMazeDistance(myPos, (width, minHeight))
        # if nextdistanceToHome < distanceToHome:
        #   reward += self.homeDistanceScaler * gameState.getAgentState(self.index).numCarrying / max(distanceToHome, 0.0000000000000000001)
        # else:
        #   reward -= self.homeDistanceScaler * gameState.getAgentState(self.index).numCarrying / max(distanceToHome, 0.0000000000000000001)
        # if gameState.data.timeleft < 200:
        #   self.foodDistanceScaler = 0
        #   self.homeDistanceScaler = 99999999999999

      
      # Distance to capsules
      capsules = self.getCapsules(gameState) 

      if len(capsules) > 0: # This should always be True,  but better safe than sorry
          myPos = nextState.getAgentState(self.index).getPosition()
          minDistanceToCapsule = min([self.getMazeDistance(nextPos, capsule) for capsule in capsules])
          reward += self.capsuleDistanceScaler / max(minDistanceToCapsule, 0.0001)

      
      # for catching invaders 
      if self.gotCaptured(gameState):
          reward += 99999999

    self.params["total_reward"][-1] += reward
    self.params["latest_reward"] = self.params["total_reward"][-1]
    self.save_weights()
    return reward

  
  
  
class QLearningDefensiveAgent(QLearningCaptureAgent):
  def __init__(self, index):
    self.param_json = 'defensiveParams_base.json'
    super().__init__(index)
  
      
      
  def getReward(self, gameState, nextState):
    """
    Get the reward for the defensive agent
    """
    reward = 0
    myPos = gameState.getAgentPosition(self.index)
    dist = self.getMazeDistance(myPos, self.startPosition)
    nextPos = nextState.getAgentState(self.index).getPosition()
    nextdist = self.getMazeDistance(nextPos, self.startPosition)

    
    enemies = [nextState.getAgentState(i) for i in self.getOpponents(nextState)]
    reward -= enemies[0].numCarrying
    reward -= enemies[1].numCarrying
    
    if self.lastPos == nextPos:
      reward -= 3
    self.lastPos = myPos
    

    if dist > 2 and nextPos == self.startPosition:
      # I die in the next state
      reward = -99999999
      self.maxHomeDistance = 0
      self.visited = set()

    if nextdist > self.maxHomeDistance and nextdist < 20 and nextPos not in self.visited:
        self.maxHomeDistance = nextdist
        reward += self.leaveHomeScaler / max(20 - dist, 0.00001)
        self.visited.add(myPos)
    else:
        reward -= self.leaveHomeScaler / max(20 - dist, 0.00001)
    # Reward for getting closer to home
    if self.color == RED:
      width = int(gameState.data.layout.width/2 - 1)
      if myPos[0] < width:
        isHome = True
      else:
        isHome = False
    else:
      width = int(gameState.data.layout.width/2 + 1)
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

    invaders = [gameState.getAgentState(i) for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman and gameState.getAgentState(i).getPosition() is not None and gameState.getAgentState(i).scaredTimer == 0]
    invaderPositions = [invader.getPosition() for invader in invaders if invader.getPosition() is not None]
  
    closestInvaderDist = 999999
    closestNextInvaderDist = 999999
    
    distsToInvaders = [self.getMazeDistance(myPos, invader) for invader in invaderPositions if invaderPositions is not None]
    distsToNextInvaders = [self.getMazeDistance(nextPos, invader) for invader in invaderPositions if invaderPositions is not None]
    if len(invaders) > 0:
      # if len(distsToInvaders) > 0 and len(distsToNextInvaders) > 0:
      maxCarrying = 0
      for i, invader in enumerate(invaders):
        # reward -= invader.numCarrying ** 2
        try:
          if invader.numCarrying > maxCarrying:
            maxCarrying = invader.numCarrying
            invaderDist = self.getMazeDistance(myPos, invaderPositions[i])
            nextInvaderDist = self.getMazeDistance(nextPos, invaderPositions[i])
        except:
          print(f"Well fuck me, here's Invader: {invader}, i: {i}, and invader positions: {invaderPositions}")
          continue
        invaderDist = min(distsToInvaders)
        nextInvaderDist = min(distsToNextInvaders)
      if nextState.getAgentState(self.index).scaredTimer == 0:
        if invaderDist > nextInvaderDist:
          reward += self.defensiveInvaderScaler / max(nextInvaderDist, 0.0001)
        else:
          reward -= self.defensiveInvaderScaler / max(nextInvaderDist, 0.0001)
      else:
        if invaderDist > nextInvaderDist + 1:
          reward += self.defensiveInvaderScaler / max(invaderDist, 0.0001)
        else:
          reward -= self.defensiveInvaderScaler / max(invaderDist, 0.0001)
    else:
      # Add negative reward for being near teammate
      teammates = self.getTeam(gameState)
      teammatePositions = [gameState.getAgentPosition(i) for i in teammates if i != self.index]
      minTeammateDist = min([self.getMazeDistance(myPos, pos) for pos in teammatePositions])
      nextMinTeammateDist = min([self.getMazeDistance(nextPos, pos) for pos in teammatePositions])
      if not isHome:
        if minTeammateDist > nextMinTeammateDist:
          reward -= self.teammateScaler / max(nextMinTeammateDist, 0.0001)
        elif minTeammateDist < nextMinTeammateDist:
          reward += self.teammateScaler / max(minTeammateDist, 0.0001)  # Scaled by closeness to teammate
      else:
        if minTeammateDist > nextMinTeammateDist:
          reward += self.teammateScaler / max(nextMinTeammateDist, 0.0001)
        elif minTeammateDist < nextMinTeammateDist:
          reward -= self.teammateScaler / max(minTeammateDist, 0.0001)  # Scaled by closeness to teammate



      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      ghosts = [ghost for ghost in enemies if not ghost.isPacman and ghost.getPosition() is not None]
      ghostPositions = [ghost.getPosition() for ghost in ghosts if ghost.getPosition() is not None]
      if len(ghosts) > 0:
        closestGhostIndex = 0
        minNextGhostDist = 999
        minGhostDist = 999
        for i, ghost in enumerate(ghosts):
          if self.getMazeDistance(myPos, ghost.getPosition()) < minGhostDist:
            minGhostDist = self.getMazeDistance(myPos, ghost.getPosition())
            closestGhostIndex = i
        minNextGhostDist = min([self.getMazeDistance(nextPos, ghost.getPosition()) for ghost in ghosts])
        closestGhostToTeammate = min([self.getMazeDistance(teammatePositions[0], ghost.getPosition()) for ghost in ghosts for teammate in teammatePositions])
        if closestGhostToTeammate < 3:
          self.capsuleDistanceScaler = 100
        else:
          self.capsuleDistanceScaler = 11
        if ghosts[closestGhostIndex].scaredTimer == 0:
          if minGhostDist > minNextGhostDist:
            reward -= self.offensiveInvaderScaler / max(minNextGhostDist, 0.0001)
          elif minGhostDist < minNextGhostDist:
            reward += self.offensiveInvaderScaler / max(minNextGhostDist, 0.0001)
        else:
          if minGhostDist > minNextGhostDist:
            reward += self.offensiveInvaderScaler / max(minNextGhostDist, 0.0001)
          elif minGhostDist < minNextGhostDist:
            reward -= self.offensiveInvaderScaler / max(minNextGhostDist, 0.0001)
        if self.isInCorner(nextPos, gameState):
          if minGhostDist > minNextGhostDist and minNextGhostDist < 8 and not isHome:
            reward -= 10 / max(minNextGhostDist, 0.0001)
          elif minGhostDist < minNextGhostDist and minNextGhostDist < 8 and not isHome:
            reward += 10 / max(minNextGhostDist, 0.0001)


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
      if distToNextFood < closestNextInvaderDist:
        if distToFood > distToNextFood:
          reward += self.foodDistanceScaler / max(distToNextFood, 0.1)
        else:
          reward -= self.foodDistanceScaler / max(distToNextFood, 0.1)
      else:
        if closestInvaderDist > closestNextInvaderDist:
          reward += self.offensiveInvaderScaler / max(closestNextInvaderDist, 0.0001)
        else:
          reward -= self.offensiveInvaderScaler / max(closestNextInvaderDist, 0.0001)

      minDistanceToHome = 9999
      minHeight = 0
      for height in range(1, gameState.data.layout.height):
        try:
          distanceToHome = self.getMazeDistance(nextPos, (width , height))
          if distanceToHome < minDistanceToHome:
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

    
      # Distance to capsules
      capsules = self.getCapsules(gameState) 

      if len(capsules) > 0: # This should always be True,  but better safe than sorry
          myPos = nextState.getAgentState(self.index).getPosition()
          minDistanceToCapsule = min([self.getMazeDistance(nextPos, capsule) for capsule in capsules])
          reward += self.capsuleDistanceScaler / max(minDistanceToCapsule, 0.0001)


      ghosts = [ghost for ghost in enemies if not ghost.isPacman and ghost.getPosition() is not None]
      if len(ghosts) > 0:
        reward += self.defensiveInvaderScaler / max(min([self.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghosts]), 0.0001)

    # for catching invaders 
    if self.gotCaptured(gameState):
        reward += 99999999999

    # Distance to capsules
    capsules = self.getCapsules(gameState) 

    if len(capsules) > 0: # This should always be True,  but better safe than sorry
        myPos = nextState.getAgentState(self.index).getPosition()
        minDistanceToCapsule = min([self.getMazeDistance(myPos, capsule) for capsule in capsules])
        minNextDistanceToCapsule = min([self.getMazeDistance(nextPos, capsule) for capsule in capsules])
        if minNextDistanceToCapsule < minDistanceToCapsule:
          reward += self.capsuleDistanceScaler / max(minDistanceToCapsule, 0.0001)
        else:
          reward -= self.capsuleDistanceScaler / max(minDistanceToCapsule, 0.0001)

    self.params["total_reward"][-1] += reward
    self.params["latest_reward"] = self.params["total_reward"][-1]
    self.save_weights()
    return reward
  
