# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
import random
import game
import util
from util import Counter
import numpy as np
from game import Grid
from collections import defaultdict


# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining=10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them

        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        # Adding and initializing the values needed for learning
        # keeps track of previous state
        self.state = None
        # keeps track of previous reward
        self.reward = None
        # keeps track of previous action
        self.action = None
        # setting the Q values table to 0 at start - it's indexed by state , action
        self.qValues = {}
        # then, setting a counter to keep count of the state, actions
        self.numStateAction = Counter()
        # adding a exploration reward
        self.exploreReward = 510
        # counting the no of visits for every state and if it doesn't, it calls the reward
        self.numState = 1
        # a boolean to initalize learning mainly for functions
        # will be false when learning is achieved.
        self.boolLearn = True
        # a boolean for first move
        self.firstMove = False
        # a boolean for the function explore method
        self.funcExplore = True
        # adjusting the alpha
        self.adjustAlpha = True
        # if it is true, then it uses the lambda alpha adjustment
        self.adjustedAlpha = lambda n: 1. / (1 + n)
        # for building network the map only once, we build the map
        self.builtMap = False
        # Training episode counter
        # counter for training the episodes
        self.lostEpiTraining = 0
        self.wonEpiTraining = 0

    # converting the moves to Integer
    def convertMoveToInteger(self, moveDirection):
        i = 0
        if moveDirection == 'East':
            i = 0
        elif moveDirection == 'South':
            i = 1
        elif moveDirection == 'West':
            i = 2
        elif moveDirection == 'North':
            i = 3
        else:
            i = 4
        return i

    # converting the moves to Integer
    def convertIntegerToMove(self, integer):
        i = 0
        if integer == 0:
            i = Directions.EAST
        elif integer == 1:
            i = Directions.SOUTH
        elif integer == 2:
            i = Directions.WEST
        elif integer == 3:
            i = Directions.NORTH
        else:
            i = Directions.STOP
        return i

    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value

    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    # Functions for Graph map and getting the wall positions

    # getWallPositions function gets the map co-ordinates of the wall
    def getWallPositions(self, state):
        positionWall = []
        xIndex = 0
        yIndex = 0
        get_walls = state.getWalls()
        for w in get_walls:
            for b in w:
                # if true, we append the indexes
                if b is True:
                    positionWall.append((xIndex, yIndex))
                yIndex += 1
            xIndex += 1
            yIndex = 0
        return positionWall

    # graphMap function is to extract dictionary to find possible paths
    def graphMap(self, positionWall):
        # adding variables for storing the x and y index and creating a defaultdict list
        xIndex = 0
        yIndex = 0
        graphMap = defaultdict(list)
        # for loop to go through the grid
        # note - since we are going for small grid, I've added the dimensions of the small grid
        for j in range(7):
            for i in range(7):
                # here, we check if the key is NOT wall and then we get the
                # right, left, up, down moves and then store them in list
                if (xIndex, yIndex) not in positionWall:
                    moveRight = (xIndex + 1, yIndex)
                    moveLeft = (xIndex - 1, yIndex)
                    moveUp = (xIndex, yIndex + 1)
                    moveDown = (xIndex, yIndex - 1)


                    pacmanDir = [moveRight, moveLeft, moveUp, moveDown]
                    # we check if the move is a wall and if not we append it to the graphMap
                    for x in pacmanDir:
                        if x not in positionWall:
                            graphMap[(xIndex, yIndex)].append(x)
                xIndex += 1
            yIndex += 1
            xIndex = 0
        return graphMap

    # Breadth First Search to find the path
    def breadthFirstSearch(self, first, goal, graph):
        store = [(first, [first])]
        while store:
            (vertex, path) = store.pop(0)
            graphVertex = set(graph[vertex])
            setPath = (set(path))

            for nextStep in graphVertex - setPath:
                # checks if our next step is the goal
                if nextStep == goal:
                    yield path + [nextStep]
                else:
                    # if not it appends the next step
                    store.append((nextStep, path + [nextStep]))

    # in key function we get the pacman position, food grid, ghost positions, the graph and state
    # PACMAN
    def key(self, pacman, food, ghost, graph, state):
        # initializing and stores the list of key representing :
        # pacman position
        pacmanList = list()
        # food position
        foodList = list()
        # ghost position
        ghostList = list()
        # state position
        stateList = list()
        # we get the pacman positions and append them to the list , we then convert it to a tuple
        for pos in pacman:
            pacmanList.append(pos)
        pacmanList = tuple(pacmanList)

        # we get the food co-ordinates and append them to the list 
        xIndex = 0
        yIndex = 0
        for f in food:
            for x in f:
                if x is True:
                    foodList.append([xIndex, yIndex])

                yIndex += 1
            xIndex += 1
            yIndex = 0
        # getting the length of the remaining food and appending that to the stateList
        remainingFood = len(foodList)
        stateList.append(remainingFood)

        # we get the ghost positions and append them to the list
        for g in ghost:
            for x in g:
                ghostList.append(x)
            # converting it to a tuple and adding it to the stateList as well
            ghostList = tuple(ghostList)
            stateList.append(ghostList)

        # we are getting the possible directions from which the ghost can come
        # Ghost
        ghostDirections = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        # find where the ghost is currently and where ghost could move next
        ghostDirection0 = list(self.breadthFirstSearch(pacmanList, ghostList, graph))[0][0]
        ghostDirection1 = list(self.breadthFirstSearch(pacmanList, ghostList, graph))[0][1]
        ghostNext = tuple(np.array(ghostDirection1) - np.array(
            ghostDirection0))

        count = 0

        # checking the ghost direction if it is following the closest distance to pacman
        for d in ghostDirections:
            if d == ghostNext:
                stateList.append(count)
            count += 1

        # finding the direction pacman should move to eat the closest food
        # Food
        foodDistance = []
        if foodList:
            for i in foodList:
                foodLen = len(list(self.breadthFirstSearch(pacmanList, tuple(i), graph))[0]) - 1
                foodDistance.append(foodLen)
        # finding the minimum food location , and calculating the direction of the first step towards it
        # via the shorest path possible
        foodIndex = foodDistance.index(min(foodDistance))
        findClosestFood = foodList[foodIndex]
        # finding the next step pacman should take to go to the food
        # via the shortest path and finding the direction
        # and then appending that direction to the movement to move closer to eat the food
        foodLocationList = list(self.breadthFirstSearch(pacmanList, tuple(findClosestFood), graph))[0][1]
        foodLocation = tuple(np.array(pacmanList) - np.array(foodLocationList))
        idx = 0
        for g in ghostDirections:
            if g == foodLocation:
                stateList.append(idx)
            idx += 1

        # we are calculating the co-ordinates for the key moves of pacman
        # State
        stateCount = 0
        up = (pacmanList[0], pacmanList[1] + 1)
        down = (pacmanList[0], pacmanList[1] - 1)
        left = (pacmanList[0] - 1, pacmanList[1])
        right = (pacmanList[0] + 1, pacmanList[1])
        stateDirections = [up, down, left, right]

        # if pacman has wall then we add +1 to counter, and in case,
        # pacman gets surrounded with 3 walls, we append 1
        walls = self.getWallPositions(state)
        for d in stateDirections:
            if d in walls:
                stateCount += 1
        if stateCount == 3:
            stateList.append(1)
        else:
        # if not 3 walls, then we append 0
            stateList.append(0)
        # we then return the statelist
        return tuple(stateList)

    # canMoveActions function to get all the legal moves
    def canMoveActions(self, legalMoves):
        canMove = []
        for l in legalMoves:
            canMove.append(self.convertMoveToInteger(l))
        return canMove

    # rewardScore function for reward signal where we subtract the current score with old score for every move
    def rewardScore(self, newScore, oldScore):
        score = float(newScore) - oldScore
        return score

    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):

        # building positions so pacman can move based on the state. Note - this is distinct for this maze
        # here we get the location of the wall
        if not self.builtMap:  # If map isn't built, build it
            self.positionWall = self.getWallPositions(state)
            self.graphMap = self.graphMap(self.positionWall)
            self.builtMap = True
        # we get the legal actions, remove stop and convert the actions that are available
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        actionsAvailable = self.canMoveActions(legal)
        self.currentState = self.key(state.getPacmanPosition(), state.getFood(), state.getGhostPositions(),
                                     self.graphMap,
                                     state)

        # here we are updating the Q values- if it is the first action we make, we skip to the else condition.
        if self.firstMove:
            # If this is the first time we have seen that state, initialize key-value pairs of all possible actions
            # to 0. If not the dictionary will be empty and not function. Allows us to add states as we see them.
            for i in actionsAvailable:
                if self.qValues.get((self.currentState, i)) == None:
                    self.qValues[self.currentState, i] = 0

            # getting the reward and then updating the old score and then we increment the s,a
            self.currentReward = self.rewardScore(state.getScore(), self.oldScore)
            self.oldScore = state.getScore()
            self.numStateAction[(self.state, self.action)] += 1
            # we calculate the alpha adjustment if it is activated or uses the existing alpha
            if self.adjustAlpha:
                self.alpha = self.adjustedAlpha(self.numStateAction[(self.state, self.action)])
            else:
                self.alpha = self.alpha
            # updating the q-table
            self.qValues[(self.state, self.action)] = self.qValues[(self.state, self.action)] + self.alpha * (
                    self.reward + self.gamma * max(self.qValues[(self.currentState, i)] for i in actionsAvailable) -
                    self.qValues[(self.state, self.action)])

        # else condition runs only in the beginning
        # old score contains the score at time 0 and we store the score in currentReward
        # we also initialize the playing state saving the bool value as True
        else:
            self.oldScore = state.getScore()
            self.currentReward = state.getScore()
            self.firstMove = True
            # we check if the dict is empty
            for i in actionsAvailable:
                if self.qValues.get((self.currentState, i)) == None:
                    self.qValues[self.currentState, i] = 0

        # to record the SCORES for every action
        self.scores = []
        # updating the state and reward
        self.state = self.currentState
        self.reward = self.currentReward
        # If using function exploration
        if self.funcExplore:
            # changing action to obtain the max rewards so we have to check all possible actions and
            # store their values and then check
            for a in actionsAvailable:
                # if the visiting state is not visited enough
                if (self.numStateAction[(self.currentState, a)] < self.numState) and self.boolLearn:
                    self.scores.append(self.exploreReward)
                # else if it has visited enough then we get the utility of that state
                else:
                    self.scores.append(self.qValues[(self.currentState, a)])

            # get the no of scores that are = to the maximum score - it'll make a random choice if there are
            # many s-a which hasn't been checked
            idxMaxScore = []
            ctr = 0
            for s in self.scores:
                if s == max(self.scores):
                    idxMaxScore.append(ctr)
                ctr += 1
            # we get the index of the max score , if there is more than one max score, we do it randomly and if only
            # only one max score then we use the first element. We map this score to the action which produced it.
            if idxMaxScore > 1:
                maxScore = random.choice(idxMaxScore)
            else:
                maxScore = idxMaxScore[0]
            self.action = actionsAvailable[maxScore]
            # converting integer action to actual action and returns the direction to move.
            moveAction = self.convertIntegerToMove(self.action)
            # returning the action from the max in q table of state,action
        return moveAction

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):

        # print "A game just ended!"
        if state.getScore() > 0:
            print "Pacman Wins!"
            print self.scores
            self.wonEpiTraining += 1
        else:
            print "Pacman died!"
            print self.scores
            self.lostEpiTraining += 1


        # updating legal state-action value. Adding reward to the state
        self.qValues[self.state, self.action] = self.qValues[self.state, self.action] + self.alpha * (
                self.reward + self.gamma * state.getScore() - self.qValues[self.state, self.action])
        # Incrementing last seen state/action pair.
        self.numStateAction[(self.state, self.action)] += 1

        # Clearing the last action reward, and state for fresh learning experience
        self.reward = None
        self.action = None
        self.state = None
        # Reinitializing boolean for starting condition
        self.firstMove = False

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        print self.getEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg, '-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)
            # displaying to check how long the pacman training is lasting
            print self.qValues

            # now that learning is done. we can set the boolean to false so the agent will select the greedy actions
            # after learning based on the q-values
            # if we leave it on true after learning, it'll continue exploration exploitation
            # ensuring we are turning off the alpha adjustment
            self.boolLearn = False
            self.adjustAlpha = False
            # Displaying the number of training episodes
            # Pacman has won & lost
            print "Training episodes Pacman won : %s \n" % self.wonEpiTraining
            print "Training episodes Pacman lost : %s \n" % self.lostEpiTraining

        # end of code
