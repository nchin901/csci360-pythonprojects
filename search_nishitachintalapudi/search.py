# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    #function tree-search(problem, strategy) returns a solution, or failure
    #initialize the search tree using the initial state of problem
    #loop do - probably a while loop
        #if there are no candidates for expansion then return failure
        #choose a leaf node for expansion according to strategy
        #if the node contains a goal state then return the corresponding solution
        #else expand the node and add the resulting nodes to the search tree
    #end
    initialState = problem.getStartState()
    fringe = util.Stack()
    visitedNodes = []

    #append -> adds to the end of the array
    #insert -> inserts element at a given index
    #mediator? variable so i can just make everything a little prettier
        #tuple that contains start state and array of visted nodes, initialization
    path = (initialState, [])
    fringe.push(path)

    #move the no candidates for expansion part out of while loop
    if problem.isGoalState(initialState):
        return visitedNodes

    #while the fringe is not empty
    #what should the conditional be? ask in oh
    while not fringe.isEmpty():
        currNode, strategy = fringe.pop()
        if currNode in visitedNodes:
            continue
        if problem.isGoalState(currNode):
            return strategy
        visitedNodes.append(currNode)
        #getsuccesors retunrns a triple so put variables for state action cost
        for successor, action, stepCost in problem.getSuccessors(currNode):
            updateStrategy = strategy+[action]
            fringe.push((successor, updateStrategy))
    return visitedNodes
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #copy pasted frm my dfs
    initialState = problem.getStartState()
    #using queue instead because i looked it up:
        #BFS uses always queue, Dfs uses Stack data structure.
        #source: https://findanyanswer.com/why-we-use-queue-in-bfs-and-stack-in-dfs#:~:text=BFS%20uses%20always%20queue%2C%20Dfs,the%20same%20idea%20as%20LIFO.
    fringe = util.Queue()
    visitedNodes = []

    path = (initialState, [])
    fringe.push(path)

    #move the no candidates for expansion part out of while loop
    if problem.isGoalState(initialState):
        return visitedNodes

    while not fringe.isEmpty():
        currNode, strategy = fringe.pop()
        if currNode in visitedNodes:
            continue
        if problem.isGoalState(currNode):
            return strategy
        visitedNodes.append(currNode)
        #getsuccesors retunrns a triple so put variables for state action cost
        for successor, action, stepCost in problem.getSuccessors(currNode):
            updateStrategy = strategy+[action]
            fringe.push((successor, updateStrategy))
    return visitedNodes
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #expand lowest path cost
    #explores options in every "direction"
    #source for explanation:
        #https://algorithmicthoughts.wordpress.com/2012/12/15/artificial-intelligence-uniform-cost-searchucs/
        #Insert the root into the queue
        #While the queue is not empty
        #Dequeue the maximum priority element from the queue
        #(If priorities are same, alphabetically smaller path is chosen)
        #If the path is ending in the goal state, print the path and exit
        #Else
            #Insert all the children of the dequeued element, with the cumulative costs as priority
    initialState = problem.getStartState()
    fringe = util.PriorityQueue()
    visitedNodes = []

    #push has self, item, priority
    #initialize queue w initial state , [] and current cost of the node
    item = initialState, [], 0
    fringe.push((item), 0)

    if problem.isGoalState(initialState):
        return visitedNodes

    #literally exactly bfs and dfs code but with priority queue instead so copy paste but cumulative costs in for loop
    while not fringe.isEmpty():
        items, num = fringe.pop()
        currNode, strategy, cost = items
        if currNode in visitedNodes:
            continue
        if problem.isGoalState(currNode):
            return strategy
        visitedNodes.append(currNode)
        #getsuccesors retunrns a triple so put variables for state action cost
        for successor, action, stepCost in problem.getSuccessors(currNode):
            updateStrategy = strategy + [action]
            costs = cost + stepCost
            fringeitem = successor, updateStrategy, costs 
            fringe.push(fringeitem, costs)
    return visitedNodes

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #copy paste ucs but account for heuristics in for loop. use heuristic() and put the childs and problem
    initialState = problem.getStartState()
    fringe = util.PriorityQueue()
    visitedNodes = []

    item = initialState, [], 0
    fringe.push(item, 0)

    if problem.isGoalState(initialState):
        return visitedNodes

    while not fringe.isEmpty():
        items, num = fringe.pop()
        currNode, strategy, cost = items
        if currNode in visitedNodes:
            continue
        if problem.isGoalState(currNode):
            return strategy
        visitedNodes.append(currNode)
        #getsuccesors retunrns a triple so put variables for state action cost
        for successor, action, stepCost in problem.getSuccessors(currNode):
            updateStrategy = strategy + [action]
            costs = cost + stepCost
            fringeitem = successor, updateStrategy, costs 
            heuristics = heuristic(successor, problem) + costs
            fringe.push(fringeitem, heuristics)
    return visitedNodes
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
