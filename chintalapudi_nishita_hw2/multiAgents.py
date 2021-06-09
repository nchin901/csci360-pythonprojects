# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util, math

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        This question is not included in project for CSCI360
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return childGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumGhost():
        Returns the total number of ghosts in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # def value(gameState, agentIndex, depth):
        #     #check for terminal states
        #     if agentIndex == gameState.getNumGhost() - 1:
        #         agentIndex = 0
        #         depth += 1
        #     if gameState.isWin() or gameState.isLose() or depth == 0:
        #         return self.evaluationFunction(gameState)

        #     #pacman's turn. agent is max
        #     if agentIndex == 0:
        #         return max_value(gameState, agentIndex, depth)
        #     #agent is min
        #     else:
        #         return min_value(gameState, agentIndex, depth)

        def max_value(gameState, agentIndex, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            v = ("", -math.inf)
            actions = gameState.getLegalActions(agentIndex)

            if len(actions) == 0:
                return self.evaluationFunction(gameState)

            for successor in actions:
                getnext = gameState.getNextState(agentIndex, successor)
                tupval = min_value(getnext, agentIndex+1, depth)
                if type(tupval) is not tuple:
                    if tupval > v[1]:
                        v = successor, tupval
                elif tupval[1] > v[1]:
                    v = (successor, tupval[1])
            return v

        def min_value(gameState, agentIndex, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
                
            v = ("", math.inf)
            actions = gameState.getLegalActions(agentIndex)

            nextLayerFunction = min_value

            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent > gameState.getNumGhost():
                nextAgent = 0
                nextDepth -= 1
                nextLayerFunction = max_value

            if len(actions) == 0:
                return self.evaluationFunction(gameState)

            for successor in actions:
                getnext = gameState.getNextState(agentIndex, successor)
                tupval = nextLayerFunction(getnext, nextAgent, nextDepth)
                if type(tupval) is not tuple:
                    if tupval < v[1]:
                        v = successor, tupval
                elif tupval[1] < v[1]:
                    v = (successor, tupval[1])
            return v
        return max_value(gameState, 0, self.depth)[0]



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(gameState, agentIndex, depth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            v = ("", -math.inf)
            actions = gameState.getLegalActions(agentIndex)

            if len(actions) == 0:
                return self.evaluationFunction(gameState)

            for successor in actions:
                getnext = gameState.getNextState(agentIndex, successor)
                tupval = min_value(getnext, agentIndex+1, depth, alpha, beta)
                if type(tupval) is not tuple:
                    if tupval > v[1]:
                        v = successor, tupval
                elif tupval[1] > v[1]:
                    v = (successor, tupval[1])
                if v[1] > beta:
                    return v[1]
                alpha = max(alpha, v[1])
            return v

        def min_value(gameState, agentIndex, depth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
                
            v = ("", math.inf)
            actions = gameState.getLegalActions(agentIndex)

            nextLayerFunction = min_value

            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent > gameState.getNumGhost():
                nextAgent = 0
                nextDepth -= 1
                nextLayerFunction = max_value

            if len(actions) == 0:
                return self.evaluationFunction(gameState)

            for successor in actions:
                getnext = gameState.getNextState(agentIndex, successor)
                tupval = nextLayerFunction(getnext, nextAgent, nextDepth, alpha, beta)
                if type(tupval) is not tuple:
                    if tupval < v[1]:
                        v = successor, tupval
                elif tupval[1] < v[1]:
                    v = (successor, tupval[1])
                if v[1] < alpha:
                    return v[1]
                beta = min(beta, v[1])
            return v

        alpha = -math.inf
        beta = math.inf
        return max_value(gameState, 0, self.depth, alpha, beta)[0]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #max value and exp value functions
        #just copy the minimax & manipulate
        #dont rly have to change the max function
        def max_value(gameState, agentIndex, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            v = ("", -math.inf)
            actions = gameState.getLegalActions(agentIndex)

            if len(actions) == 0:
                return self.evaluationFunction(gameState)

            for successor in actions:
                getnext = gameState.getNextState(agentIndex, successor)
                tupval = exp_value(getnext, agentIndex+1, depth)
                if type(tupval) is not tuple:
                    if tupval > v[1]:
                        v = successor, tupval
                elif tupval[1] > v[1]:
                    v = (successor, tupval[1])
            return v

        def exp_value(gameState, agentIndex, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
                
            #v = ("", math.inf)
            v = ("", 0)
            actions = gameState.getLegalActions(agentIndex)
            p = 1.0 / len(actions)

            nextLayerFunction = exp_value

            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent > gameState.getNumGhost():
                nextAgent = 0
                nextDepth -= 1
                nextLayerFunction = max_value

            if len(actions) == 0:
                return self.evaluationFunction(gameState)

            for successor in actions:
                getnext = gameState.getNextState(agentIndex, successor)
                tupval = nextLayerFunction(getnext, nextAgent, nextDepth)
                # if type(tupval) is not tuple:
                #     if tupval < v[1]:
                #         v = successor, tupval
                # elif tupval[1] < v[1]:
                #     v = (successor, tupval[1])
                #p = probability of successor
                if type(tupval) is not tuple:
                    #p = tupval / len(actions)
                    v = successor, ((tupval * p) + v[1])
                else:
                    v = successor, ((tupval[1] * p) + v[1])
            return v
        return max_value(gameState, 0, self.depth)[0]



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 4).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #evaluate states instead of actions
    #this is literally the vaguest set of instructions i have ever seen < 3
    #old evaluation function
    # childGameState = currentGameState.getPacmanNextState(action)
    # newPos = childGameState.getPacmanPosition()
    # newFood = childGameState.getFood()
    # newGhostStates = childGameState.getGhostStates()
    # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    ghostdist = math.inf
    capsuledist = math.inf
    scaredghostlist = 0

    currentposition = currentGameState.getPacmanPosition()
    # currentfood = currentGameState.getFood().asList()
    currentfoodcount = currentGameState.getNumFood()
    currentcapsules = currentGameState.getCapsules()
    currentghoststate = currentGameState.getGhostStates()
    currentscaredtimes = [ghostState.scaredTimer for ghostState in currentghoststate]

    food = 1.0 / (currentfoodcount + 1.0)

    #for the food in the current food inthe list find the manhattan distance betewen food & pacman's position
    # for food in currentfood:
    #     foodlist += manhattanDistance(food, currentposition)
    # fdistancemin = min(foodlist)

    #for ghost in ghostlist account for scared or not and find manhattan distance between ghost and pacman
    for ghost in currentghoststate:
        gposition = ghost.getPosition()
        if currentposition == gposition:
            return -math.inf
        else:
            gmandist = manhattanDistance(currentposition, gposition)
            ghostdist = min(ghostdist, gmandist)
        if ghost.scaredTimer != 0:
            scaredghostlist += 1
    ghostdist = 1.0 / (1.0 + (ghostdist/ len(currentghoststate)))
    scaredghostlist = 1.0 / (1.0 + scaredghostlist)

    for capsule in currentcapsules:
        cmandist = manhattanDistance(currentposition, capsule)
        capsuledist = min(capsuledist, cmandist)
    capsuledist = 1.0 / (1.0 + len(currentcapsules))

    score = currentGameState.getScore() + (food + ghostdist + capsuledist)
    return score


# Abbreviation
better = betterEvaluationFunction
