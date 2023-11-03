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
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodList = currentGameState.getFood().asList()
        nextFoodList = successorGameState.getFood().asList()
        capsuleList = currentGameState.getCapsules()
        x, y = newPos
        score = successorGameState.getScore()

        if successorGameState.isWin():
            return float('inf')
        if successorGameState.hasWall(x, y):
            return -2500
        # Resta puntos con base en la distancia mas cercana hacia la comida
        stepsToFood = []
        for food in foodList:
            foodX, foodY = food
            stepsAway = abs(foodX - x) + abs(foodY - y)
            stepsToFood.append(stepsAway)
        score -= min(stepsToFood) * 2
        # Resta puntos con base en la distancia mas cercana hacia una fantasma
        stepsToGhosts = []
        for ghost in range(len(newGhostStates)):
            ghostX, ghostY = newGhostStates[ghost].getPosition()
            stepsAway = abs(ghostX - x) + abs(ghostY - y)
            stepsToGhosts.append(stepsAway)
        if min(stepsToGhosts) == 0:
            return float('-inf')
        score += min(stepsToGhosts)*2

        # Resta puntos con base en la distancia mas cercana hacia una capsula, si
        # es que hay una
        if capsuleList:
            stepsToCap = []
            for cap in capsuleList:
                capX, capY = cap
                stepsAway = abs(capX - x) + abs(capY - y)
                stepsToCap.append(stepsAway)
            score -= min(stepsToCap) * 2
        #    Si pacman come con esa accion, se le suman 5 puntos al puntaje
        if len(nextFoodList) < len(foodList):
            score += 5
        # Si la accion es detenerse, se restan 10 puntos
        if action == Directions.STOP:
            score -= 10


        return score



def scoreEvaluationFunction(currentGameState: GameState):
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
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # howManyGhosts = gameState.getNumAgents()

        # Get the 4 legal actions
        # Generate a successor with each one of them
        # Evaluate every one
        # Take the one with the biggest reward

        actionIndex = 0
        # Iniciamos el valor de la recompensa en - infinito
        currentMaxValue = -float('inf')
        # Obtenemos las acciones posibles para pacman (agente 0)
        legalActions = gameState.getLegalActions(0)
        # Obtenemos los estados sucesores para cada accion
        for i in range(len(legalActions)):
            nextStep = gameState.generateSuccessor(0, legalActions[i])
            # Obtenemos la utilidad terminal de cada estado sucesor
            reward = self.value(nextStep, 1, 0)
            # Si la recompensa es mayor a la actual, la guardamos
            if reward > currentMaxValue:
                currentMaxValue = reward
                actionIndex = i

        return legalActions[actionIndex]

    # Funcion MIN de minimax
    def minValue(self, gameState: GameState, agentIndex, currentDepth):
        # Obtenemos las acciones posibles del jugador (agente)
        legalActions = gameState.getLegalActions(agentIndex)
        succesors = []
        # Iniciamos v en + infinito
        v = float('inf')
        # Obtenemos los estados sucesores para cada accion
        for legalAction in legalActions:
            succesors.append(gameState.generateSuccessor(agentIndex, legalAction))
        for s in succesors:
            # Si el siguiente agente(fantasma) es el ultimo, evaluamos a pacman de nuevo
            if (agentIndex + 1) == gameState.getNumAgents():
                v = min(v, self.value(s, 0, currentDepth+1))
            #     Si no, pasamos al siguiente agente(fantasma)
            else:
                v = min(v, self.value(s, agentIndex + 1, currentDepth))
        return v

    # Funcion MAX de minimax (solo para pacman)
    def maxValue(self, gameState: GameState,  currentDepth ):
        # Obtenemos las acciones de pacman
        legalActions = gameState.getLegalActions(0)
        successors = []
        # Iniciamos v en - infinito
        v = -float('inf')
        # Obtenemos los estados sucesores para cada accion
        for action in legalActions:
            successors.append(gameState.generateSuccessor(0, action))
        for s in successors:
            # Obtenemos la recompensa mas grande del agente 1
            v = max(v, self.value(s, 1, currentDepth))
        return v

    # Funcion evaluadora
    def value(self, gameState: GameState, agentIndex, currentDepth):
        # Si el nivel que esta siendo evaluado es igual al limite de profundidad establecido
        # O el estado es ganador o perdedor
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            # Se regresa la utilidad terminal
            return self.evaluationFunction(gameState)
        # Si el agente es pacman
        if agentIndex == 0:
            # Se llama a la funcion MAX
            return self.maxValue(gameState, currentDepth)
        # Si es un fantasma
        if agentIndex >= 1:
            # Se llama a la funcion MIN
            return self.minValue(gameState, agentIndex, currentDepth)
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actionIndex = 0
        # la mejor recompensa para MAX
        alpha = -float('inf')
        # la mejor recompensa para MIN
        beta = float('inf')
        # Iniciamos el valor de la recompensa en - infinito
        currentMaxValue = -float('inf')
        # Obtenemos las acciones posibles para pacman (agente 0)
        legalActions = gameState.getLegalActions(0)
        # Obtenemos los estados sucesores para cada accion
        for i in range(len(legalActions)):
            nextStep = gameState.generateSuccessor(0, legalActions[i])
            # Obtenemos la utilidad terminal de cada estado sucesor
            reward = self.value(nextStep, 1, 0, alpha, beta)
            # Si la recompensa es mayor a la actual, la guardamos
            if reward > currentMaxValue:
                currentMaxValue = reward
                alpha = reward
                actionIndex = i

        return legalActions[actionIndex]


    # Funcion MIN de minimax
    def minValue(self, gameState: GameState, agentIndex, currentDepth, alpha, beta):
        # Obtenemos las acciones posibles del jugador (agente)
        legalActions = gameState.getLegalActions(agentIndex)
        # Iniciamos v en + infinito
        v = float('inf')
        # Obtenemos los estados sucesores para cada accion
        for legalAction in legalActions:
            succesor = (gameState.generateSuccessor(agentIndex, legalAction))
            # Si el siguiente agente(fantasma) es el ultimo, evaluamos a pacman de nuevo
            if (agentIndex + 1) == gameState.getNumAgents():
                v = min(v, self.value(succesor, 0, currentDepth + 1, alpha, beta))
            #     Si no, pasamos al siguiente agente(fantasma)
            else:
                v = min(v, self.value(succesor, agentIndex + 1, currentDepth, alpha, beta))
            if v < alpha:
                break
            beta = min(beta, v)
        return v


    # Funcion MAX de minimax (solo para pacman)
    def maxValue(self, gameState: GameState, currentDepth, alpha, beta):
        # Obtenemos las acciones de pacman
        legalActions = gameState.getLegalActions(0)
        # Iniciamos v en - infinito
        v = -float('inf')
        # Obtenemos los estados sucesores para cada accion
        for action in legalActions:
            successor = (gameState.generateSuccessor(0, action))
            # Obtenemos la recompensa mas grande del agente 1
            v = max(v, self.value(successor, 1, currentDepth, alpha, beta))
            if v > beta:
                break
            alpha = max(alpha, v)
        return v


    # Funcion evaluadora
    def value(self, gameState: GameState, agentIndex, currentDepth, alpha, beta):
        # Si el nivel que esta siendo evaluado es igual al limite de profundidad establecido
        # O el estado es ganador o perdedor
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            # Se regresa la utilidad terminal
            return self.evaluationFunction(gameState)
        # Si el agente es pacman
        if agentIndex == 0:
            # Se llama a la funcion MAX
            return self.maxValue(gameState, currentDepth, alpha, beta)
        # Si es un fantasma
        if agentIndex >= 1:
            # Se llama a la funcion MIN
            return self.minValue(gameState, agentIndex, currentDepth, alpha, beta)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        actionIndex = 0
        # Iniciamos el valor de la recompensa en - infinito
        currentMaxValue = -float('inf')
        # Obtenemos las acciones posibles para pacman (agente 0)
        legalActions = gameState.getLegalActions(0)
        # Obtenemos los estados sucesores para cada accion
        for i in range(len(legalActions)):
            nextStep = gameState.generateSuccessor(0, legalActions[i])
            # Obtenemos la utilidad terminal de cada estado sucesor
            reward = self.value(nextStep, 1, 0)
            # Si la recompensa es mayor a la actual, la guardamos
            if reward > currentMaxValue:
                currentMaxValue = reward
                actionIndex = i

        return legalActions[actionIndex]

    # Funcion EXP de minimax
    def expValue(self, gameState: GameState, agentIndex, currentDepth):
        # Obtenemos las acciones posibles del jugador (agente)
        legalActions = gameState.getLegalActions(agentIndex)
        # Iniciamos v en + infinito
        v = 0.0
        # Obtenemos los estados sucesores para cada accion
        for legalAction in legalActions:
            succesor = (gameState.generateSuccessor(agentIndex, legalAction))
            # Si el siguiente agente(fantasma) es el ultimo, evaluamos a pacman de nuevo
            if (agentIndex + 1) == gameState.getNumAgents():
                # Asumimos que la probabilidad de cada sucesor es igual para cada uno,
                # por eso se multiplica por 1/total de estados sucesores
                successorValue = self.value(succesor, 0, currentDepth + 1)
                v += successorValue*(1/len(legalActions))
            #     Si no, pasamos al siguiente agente(fantasma)
            else:
                successorValue = self.value(succesor, agentIndex + 1, currentDepth)
                v += successorValue*(1/len(legalActions))
        return v

    # Funcion MAX de minimax (solo para pacman)
    def maxValue(self, gameState: GameState, currentDepth):
        # Obtenemos las acciones de pacman
        legalActions = gameState.getLegalActions(0)
        # Iniciamos v en - infinito
        v = -float('inf')
        # Obtenemos los estados sucesores para cada accion
        for action in legalActions:
            successor = (gameState.generateSuccessor(0, action))
            # Obtenemos la recompensa mas grande del agente 1
            v = max(v, self.value(successor, 1, currentDepth))
        return v

    # Funcion value
    def value(self, gameState: GameState, agentIndex, currentDepth):
        # Si el nivel que esta siendo evaluado es igual al limite de profundidad establecido
        # O el estado es ganador o perdedor
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            # Se regresa la utilidad terminal
            return self.evaluationFunction(gameState)
        # Si el agente es pacman
        if agentIndex == 0:
            # Se llama a la funcion MAX
            return self.maxValue(gameState, currentDepth)
        # Si es un fantasma
        if agentIndex >= 1:
            # Se llama a la funcion MIN
            return self.expValue(gameState, agentIndex, currentDepth)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    foodList = currentGameState.getFood().asList()
    currentPos = currentGameState.getPacmanPosition()
    capsuleList = currentGameState.getCapsules()
    x, y = currentPos
    score = scoreEvaluationFunction(currentGameState)
    if currentGameState.isWin():
        return 10000000
    elif currentGameState.isLose():
        return -10000000
    if currentGameState.hasWall(x, y):
        return -2500
    stepsToFood = []
    # Resta puntos con base en la distancia mas cercana hacia la comida
    for food in foodList:
        foodX, foodY = food
        stepsAway = abs(foodX - x) + abs(foodY - y)
        stepsToFood.append(stepsAway)
    score -= min(stepsToFood)*2
    # Resta puntos con base en la cantidad de comida restante
    score -= len(foodList)*4
    # Resta puntos con base en la cantidad de capsulas restantes
    score -= len(capsuleList)*4
    return score
# Abbreviation
better = betterEvaluationFunction
