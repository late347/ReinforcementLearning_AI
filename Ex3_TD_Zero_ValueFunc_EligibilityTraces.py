import numpy as np
import numpy.linalg as LA
import random
from datetime import datetime
from numpy import random as npRAND
import time

random.seed(datetime.now())
npRAND.seed(int(time.time()))

actDict={0:"U",1:"R",2:"D",3:"L"}
reward = -1
rows_count = 4
columns_count = 4
maxiters = 5000
##maxiters = 100000
alpha = 0.001
gamma = 1.0
lambda_factor = 0.95

V = np.zeros((rows_count, columns_count))
policies = np.array([ ['T','A','A','A'],
                     ['A','A','A','A'],
                     ['A','A','A','A'],
                     ['A','A','A','T'] ])



E_trace = np.array([ [0.0,0.0,0.0,0.0],
                     [0.0,0.0,0.0,0.0],
                     [0.0,0.0,0.0,0.0],
                     [0.0,0.0,0.0,0.0]])



def isTerminal(r, c):  # helper function to check if terminal state or regular state
    global rows_count, columns_count
    if r == 0 and c == 0:  # im a bit too lazy to check otherwise the iteration boundaries
        return True  # so that this helper function is a quick way to exclude computations
    if r == rows_count - 1 and c == columns_count - 1:
        return True
    return False


def getValue(row, col):  # helper func, get state value
    global V
    if row == -1:
        row = 0  # if you bump into wall, you bounce back
    elif row == 4:
        row = 3
    if col == -1:
        col = 0
    elif col == 4:
        col = 3
    return V[row, col]


def getRandomStartState():
    illegalState = True
    while illegalState:
        r = random.randint(0, 3)
        c = random.randint(0, 3)
        if isTerminal(r,c):
            illegalState = True
        else:
            illegalState = False
    return r, c


def getState(row, col):
    global rows_count, columns_count
    if row < 0:
        row = 0  # helper func for the exercise:1
    elif row > rows_count-1:
        row = 3
    if col < 0:
        col = 0
    elif col > columns_count-1:
        col = 3
    return row, col


def getRandomWalkAction():
    actions = ["U", "R", "D", "L"]
    probs = [0.25, 0.25, 0.25, 0.25]
    chosenAct = npRAND.choice( actions, 1, p=probs )
    return str(chosenAct[0])

def getRandomAction():
    global actDict
    return actDict[random.randint(0, 3)]


def TD_Zero_Episode( r, c, act ):
    global reward
    stepsTaken = 0
    curR = r
    curC = c
    episodeList = [ ((r, c), act, reward) ]  # add the starting (s,a,r) immediately

    if act == "U":  ##up
        r -= 1
    elif act == "R":  ##right
        c += 1
    elif act == "D":  ## down
        r += 1
    elif act == "L":  ##left
        c -= 1
    stepsTaken += 1
    r, c = getState(r, c)  ## check status of the newState (s')
    stateWasTerm = isTerminal(r, c)  ## if status was terminal stop iteration, else keep going into loop

    if not stateWasTerm:
        curR = r
        curC = c
    else:
        curR = r
        curC = c
        episodeList.append(((curR, curC), act, reward))  ## put the final state into episode

    while not stateWasTerm:
        act = getRandomAction() ## get action from randomwalk
        if act == "U":  ## up
            r -= 1
        elif act == "R":  ## right
            c += 1
        elif act == "D":  ## down
            r += 1
        elif act == "L":  ## left
            c -= 1
        stepsTaken += 1

        r, c = getState(r, c) ## check where newState will be, if we hit the wall and bounce back
        stateWasTerm = isTerminal(r, c)  ## was the newState terminalState or not
        episodeList.append( ((curR, curC), act, reward) ) ## put currentState into episodeList
        if not stateWasTerm:
            curR = r    ## if newState was not terminalSTate => update currentState = newState
            curC = c
        else:
            curR=r
            curC=c
            episodeList.append( ((curR, curC), act, reward) ) ## put the final state into episode
        if stepsTaken >= 100000:   ## if we get stuck in foreverloop, for episode generation, raise Exception
            raise Exception("Exception raised, because program got stuck in MC Qepisode generation...\n")

    return episodeList



def makeAction( S_t, A_t, ):
    global reward
    r = S_t[0]
    c = S_t[1]
    stateWasTerm = False
    if A_t == "U":  ## up
        r -= 1
    elif A_t == "R":  ## right
        c += 1
    elif A_t == "D":  ## down
        r += 1
    elif A_t == "L":  ## left
        c -= 1
    else:
        raise Exception("illegalACtion as parameter, in makeAction()")
    r,c = getState(r,c)
    stateWasTerm = isTerminal(r,c)
    return (r,c), reward, stateWasTerm


def updateE_trace(visitedState):
    global E_trace, rows_count, columns_count
    global gamma, lambda_factor
    r = visitedState[0]
    c = visitedState[1]
    tempVal = E_trace[r, c]
    tempVal += 1
    E_trace[r, c] = tempVal

    ## NOTE call this updateE_trace func only before you have updated the S= Sprime
    ## such that when you do it that way, then you wont run into problems updating the
    ## terminalStates stateval
    for row in range(rows_count):
        for col in range(columns_count):
            decayedVal = E_trace[row, col]
            decayedVal = decayedVal * gamma * lambda_factor
            E_trace[row, col] = decayedVal

def nullifyE_trace():
    global E_trace
    E_trace = np.array([[0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0]])

for iteration in range(1, maxiters+1):
    print("TD(zero)_ValFunc_iterCount == ", iteration)
    for curR in range(rows_count):
        for curC in range(columns_count):
            if not isTerminal(curR,curC):
                nullifyE_trace()
                ## EPISODE STARTS
                S = (curR, curC)
                while not isTerminal(S[0], S[1]):
                    A = getRandomAction()
                    r = S[0]
                    c = S[1]
                    Sprime, R, endedInTerminalState = makeAction(S, A)
                    r_1 = Sprime[0]
                    c_1 = Sprime[1]
                    curValue = V[ r, c ]
                    newValue = V[ r_1, c_1 ]
                    V[ r, c ] = (curValue + alpha * ( R + gamma * newValue- curValue ))
                    updateE_trace(S) ## updates the Etrace for visitedState and non-visited states

                    S = Sprime ## updates curState = newState



print("TD(Zero) ValueFunc was \n\n", V)

print("E_trace func was \n\n", E_trace)