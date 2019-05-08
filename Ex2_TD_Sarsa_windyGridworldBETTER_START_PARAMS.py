import numpy as np
import numpy.linalg as LA
import random
from datetime import datetime
from numpy import random as npRAND
import time

"""provide randomSeeds ffor numpy.random and python random"""
random.seed(datetime.now())
npRAND.seed(int(time.time()))

rows_count = 7
columns_count = 10
start_row =3
start_col=0
term_row=3
term_col=7

start_state = (start_row,start_col)
term_state = (term_row,term_col)


def isTerminal(r, c):  # helper function to check if terminal state or regular state
    global term_state
    wasTerm = False
    if r == term_state[0] and c == term_state[1]:
        wasTerm = True

    return wasTerm



"""STARTING VARIABLES AND PARAMETERS"""

"""note that the book parameters for windyWorld were as follows
alpha = 0.5
eps = 0.1
maxiters = 10000
possibly not even an eps-reducing schedule


note  that my own testing found out that these parameters worked better
for optimal windyWorld Qfunc (I think it's not totally guaranteed, sometimes it's one stepp un-optimal?)
alpha = 0.20
eps = 0.5
maxiters = 15000
eps-reducing schedule was:
    every iteration, eps=eps*0.999

These my own parameters worked good, because larger starting epsilon guaranteed
that early on there was more exploration.
Also smaller alpha meant that bad policies are forgotten more quickly???!!!
Epsilon-reducing schedule guaranteed that eploitation becomes much more frequent in midphase and latephase
"""

reward = -1
maxiters = 10000
alpha = 0.20
gamma = 1.0
reduceEpsFactor = 0.999
V = np.zeros((rows_count, columns_count))
actDict={0:"U",1:"R",2:"D",3:"L"}
epsilon = 0.5
constEpsilon = epsilon ## original epsilon was
actions = ["U", "R", "D", "L"]
returnsDict={}
QDict={}
actDict={0:"U",1:"R",2:"D",3:"L"}
policies = {}
probDist = [0.25, 0.25, 0.25, 0.25] ## up,right,down,left, Starts with equiprobable randomwalk policy for all (s)


"""printedpolicy is for printing at the end to check if algorithm
was optimal or how close to it, it was or wasnt
(you check if by looping thru the policy and choosign argmax action always)"""
printedPolicy = np.array([  ['A','A','A','A','A','A','A','A','A','A'],
                            ['A','A','A','A','A','A','A','A','A','A'],
                            ['A','A','A','A','A','A','A','A','A','A'],
                            ['A','A','A','A','A','A','A','T','A','A'],
                            ['A','A','A','A','A','A','A','A','A','A'],
                            ['A','A','A','A','A','A','A','A','A','A'],
                            ['A','A','A','A','A','A','A','A','A','A'] ])

"""we, can test the final Qfunction and policy on an empty board, by
walking inside of that board with our Qfunction and choosing always argmaxAct
we can update our walked path into the board, to see if we walked optimally
we will also take the windforce into account as needed 
(e.g. when windforcevalue >= 1, 
then we will print diagonal move into the startingState
then, newState will be the diagonal state where the arrow points to)"""
emptyBoard = np.array([  ['o','o','o','o','o','o','o','o','o','o'],
                            ['o','o','o','o','o','o','o','o','o','o'],
                            ['o','o','o','o','o','o','o','o','o','o'],
                            ['o','o','o','o','o','o','o','T','o','o'],
                            ['o','o','o','o','o','o','o','o','o','o'],
                            ['o','o','o','o','o','o','o','o','o','o'],
                            ['o','o','o','o','o','o','o','o','o','o'] ])


print("printedPolicy at start was  \n\n",printedPolicy)








"""returnsDict, for each state-action pair, maintain (mean,visitedCount)"""
for r in range(rows_count):
    for c in range(columns_count):
        if not isTerminal(r, c):
            for act in actions:
                returnsDict[ ((r, c), act) ] = [0, 0] ## Maintain Mean, and VisitedCount for each state-action pair


"""policy, now contains action distribution for possible actions"""
for r in range(rows_count):
    for c in range(columns_count):
        policies[(r,c)] = probDist.copy()  ## each state, has it's own action distribution, which sounds about right...



""" Qfunc, we maintain the action-value for each state-action pair"""
for r in range(rows_count):
    for c in range(columns_count):
        for act in actions:
            if not isTerminal(r,c):
                QDict[ ((r,c), act) ] = 0  ## initialize non-terminal (s,a) pairs arbitrarily
            else:
                QDict[  ((r,c), act) ] = 0  ## for terminal (s,a) pairs we must initialize at zero according to Sutton & Barto




"""parameter state tuple
returns the chosenACtion from epsilonGreedy policy (string action_0)
NOTE!!! IS NOT GUARANTEED TO RETURN optimal action!!! """
def getActionFromEpsGreedyPolicy(S_t):
    global policies, actions
    curActDistr = policies[ S_t ].copy()
    chosenAct = npRAND.choice( actions, 1, p=curActDistr )
    chosenAct = str(chosenAct[0])
    return chosenAct


"""parameters state tuple, currentOptimalAction (argmaxAct)
void return,
updates the global policies action distribution"""
def updateEpsGreedyPolicy( S_t, argmaxAction):
    global policies, epsilon, actions
    curGreedyAct = argmaxAction
    greedyProb = (1-epsilon +epsilon/4.0) ## P(A) ==  1-eps + eps/4
    rndProb = (1-greedyProb)/3.0 ## P(B) ==  1-P(A) / 3, because it will be three other actions, plus one greedy action
    ind = 0
    newProbs = [0, 0, 0, 0]

    for act in actions:
        if act == "U":
            ind = 0
        elif act == "R":
            ind = 1
        elif act == "D":
            ind = 2
        elif act == "L":
            ind = 3
        if act == curGreedyAct:
            newProbs[ind] = greedyProb
        else:
            newProbs[ind] = rndProb

    policies[ S_t ] = newProbs.copy()



"""get random starting state
returns row,col """
def getRandomStartState():
    global start_state
    illegalState = True
    while illegalState:
        r = random.randint(0, rows_count-1)
        c = random.randint(0, columns_count-1)
        if ( isTerminal( r, c) ):
            illegalState = True
        else:
            illegalState = False
    return r, c

"""get state from array, based on 4x4 gridworld
will prevent going outside bounds
returns row,col"""
def getState(row, col):
    global rows_count, columns_count
    if row < 0:
        row = 0
    elif row > rows_count-1:
        row = rows_count-1
    if col < 0:
        col = 0
    elif col > columns_count-1:
        col = 3
    return row, col

"""get random equiprobably action from 4 actions(u,r,d,l)
returns the random action 'U' string"""
def getRandomAction():
    global actDict
    return actDict[random.randint(0, 3)]



"""compute the mean incrementally based on visitedCount, oldMean, and currentReturnG
returns the newMean float"""
def getMeanFromReturns(oldMean, n, curVal):
    if n == 0:
        raise Exception('Exception, incrementalMeanFunc, n should not be less than 1')
    elif n == 1:
        return curVal
    elif n >= 2:
        newMean = (float) ( oldMean + (1.0 / n) * (curVal - oldMean) )
        return newMean


"""makes the action from startinState, with startingAct
e.g. (2,2) and 'U'
parameters stateTuple, actionString
returns newState, reward, boolNewStateWasTerminal"""
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


def makeWindyWorldAction( S_t, A_t, ):
    global reward
    startingState = S_t

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

    """after moving to the 'newState' in windyworld,
    we must check the windforce values from the original startingState
    if windforce == 0, move regularly into the newState and check terminalState
     
     Othewise, windforce moves the newState into another newState, northwards, based on windforcevalue,
     and then you check if it was terminalState
     NOTE! because windforce is northwards, we just decrement r variable (goes up)"""

    windforcevalue = getWindyStrength(startingState)
    if windforcevalue == 0:
        stateWasTerm = isTerminal(r,c)
        return (r,c), reward, stateWasTerm
    else:
        r -= windforcevalue
        r,c = getState(r,c)
        stateWasTerm = isTerminal(r,c)
        return (r,c), reward, stateWasTerm





"""gets the windstrength for startingState S_t
parameter S_t
returns the windstrength in amount of squares you move northwards"""
def getWindyStrength(S_t):
    startingCol = S_t[1]
    windForceList = [0,0,0,1,1,1,2,2,1,0]
    windstrength = windForceList[ startingCol ]
    return windstrength



"""get the best action 
returns string action e.g. 'D'
parameter is state tuple (r,c)"""
def getArgmaxActQ(S_t):
    global QDict
    qvalList = [  ]
    saList = [(S_t, "U"),  (S_t, "R"), (S_t, "D"), (S_t, "L") ]
    """for example get together
    s1a1, s1a2, s1a3, s1a4
    find which is the maxValue, and get the action which caused it"""
    q1 = QDict[saList[0]]
    q2 = QDict[saList[1]]
    q3 = QDict[saList[2]]
    q4 = QDict[saList[3]]
    qvalList.append(q1)
    qvalList.append(q2)
    qvalList.append(q3)
    qvalList.append(q4)
    maxQ = max(qvalList)
    ind_maxQ = qvalList.index( maxQ )   # gets the maxQ value and the index which caused it
    """when we have index of maxQval, then we know which sa-pair
    gave that maxQval => we can access that action from the correct sa-pair"""
    argmaxAct = saList[ind_maxQ][1]
    return argmaxAct



def refactored_argmaxActQ(S_t):
    global QDict, actions
    """purpose of the refactored function
    optAct = argmaxAct_Q(S_t)
    gets the action, that which gives maximum value action-value from that state S_t
    """
    state_act_pairs = [  ( (S_t), act )    for act in actions ]
    q_values = [ QDict[ x ] for x in state_act_pairs ]
    ind_max_q = q_values.index( max( q_values ) )
    argmaxAct = state_act_pairs[ ind_max_q ][1]
    return argmaxAct




def getWalkedPathSymbol(S_0, S_1):
    upLeft = '↖'
    upRight = '↗'
    downRight = '↘'
    downLeft = '↙'
    up = '↑'
    right = '→'
    down = '↓'
    left = '←'
    symbol = left

    r0 = S_0[0]
    c0 =S_0[1]
    r1=S_1[0]
    c1=S_1[1]

    if (r1 != r0 and c1 != c0):  ##diagonal moves are here
        ##first up
        if r0-r1 == 1:
            if c1-c0 == 1:
                symbol = upRight
            elif c0-c1 == 1:
                symbol = upLeft
            else:
                raise Exception("walkedPath updir bug")
        ##down
        elif r1-r0 == 1:
            if c1-c0 == 1:
                symbol = downRight
            elif c0-c1 == 1:
                symbol = downLeft
            else:
                raise Exception("walkedPath downdir bug")
        else:
            raise Exception("walkedPath diagonaldir bug")




    else:  ## straight moves are here
        if r0 == r1: ##horiz
            if c1 - c0 == 1: ## rightmove here
                symbol = right
            elif c0 -c1 == 1: ## leftmove here
                symbol = left
            else:
                raise Exception("walkedPath horizdir bug")

        elif c0 == c1: ##vert
            if r1-r0 == 1:
                symbol = down
            elif r0-r1 == 1:
                symbol = up
            else:
                raise Exception("walkedPath vertdir bug")
        else:
            raise Exception("walkedPath straightmove bug")


    return symbol




print("hello TD_SARSA, WindyGridWorld!!!")


for iteration in range(1, maxiters+1):
    print("TD_SARSA_WindyWorld iterCount == ", iteration, "eps == ", epsilon , end =" ")

    ##if iteration % 20 == 0: ## get random seed periodically to improve randomness performance in episodeGeneration
      ##  random.seed(datetime.now())
    ##if iteration % 20 == 0:

    """episode begins in earnest! now"""
    S = start_state  ## get startState and startAction
    ##A = getActionFromEpsGreedyPolicy(S)
    ##optA = getArgmaxActQ(S)  ## NOTE!!! important! we must explicitly get argmaxAct, because getActFromEpsGreedy is not guaranteed to return optAct!!!
    ##updateEpsGreedyPolicy(S, optA)  ## update eps-greedy policy with current (S,Aopt) probabilities,
    stepsEpisode = 0
    while not isTerminal(S[0], S[1]):
        A = getActionFromEpsGreedyPolicy(S)
        optA = getArgmaxActQ(S)  ## NOTE!!! important! we must explicitly get argmaxAct, because getActFromEpsGreedy is not guaranteed to return optAct!!!
        updateEpsGreedyPolicy(S, optA)  ## update eps-greedy policy with current (S,Aopt) probabilities,
        stepsEpisode += 1
        r0 = S[0]   ## simply decompose curState into components
        c0 = S[1]
        Sprime, R, endedInTerminalState = makeWindyWorldAction(S, A) ## make windyWorldAction, if there was wind => Sprime will be with windCorrection included
        Aprime = getActionFromEpsGreedyPolicy(Sprime)
        r1 = Sprime[0]    ## simple decompose newState into components
        c1 = Sprime[1]
        curValue = QDict[ S, A ]
        newValue = QDict[ Sprime, Aprime ]
        QDict[ (S, A) ] = (curValue + alpha * ( R + gamma * newValue- curValue ))
        S = Sprime
        A = Aprime
        ##optAprime = getArgmaxActQ(Sprime) ## NOTE!!! important! we must explicitly get argmaxAct, because getActFromEpsGreedy is not guaranteed to return optAct!!!
        ##updateEpsGreedyPolicy( Sprime, optAprime ) ## update eps-greedy policy with new (Sprime,AprimeOpt)probabilities
    epsilon = epsilon * reduceEpsFactor  ## reduce epsilon, favor the exploration near start, and favor exploitation near end
    print(", stepsEpisode == ", stepsEpisode)

print("TD_Sarsa Qfunc0 was \n\n", QDict)



"""print the epsGreedyPolicy"""
print("\n the eps greedy optimal policy was \n\n")
for SApair in QDict:
    if   not isTerminal( SApair[0][0],SApair[0][1] ) and ( QDict[SApair] != 0 ):
        curState = SApair[0]
        optA = getArgmaxActQ( curState )
        r0 = curState[0]
        c0 = curState[1]
        printedPolicy[r0,c0] = optA
print(printedPolicy)


"""Walking the path of the righteous in the valley of Wind...

how do we prove, that our Qfunction and policy will be both optimal ones?
We should start on an empty board now...
start in curState =  (3,0)
with the current state, get the argmax[ Q(S) ], this shoiuld tell us which action is regarded best, Note we just read from the Qfunction, no longer improve it,
make that argmaxAct in the empty board, and check where the windforce puts you in the empty board 
update curState = newState
break if curState == termState"""

print('\n supposedly the optimal path across board will be \n')

gotStuckCounter = 0
curState = start_state
while curState != term_state:
    gotStuckCounter += 1
    bestAct = getArgmaxActQ(curState)
    newState, reward, wasNewTerminalState = makeWindyWorldAction(curState, bestAct) ## using the good-old MakeWindyWorldAction function just for debug purposes!!!
                                                                                    ## you dont want to change too many things at the same time!!!
    sym = getWalkedPathSymbol(curState, newState)  ## testing the new fucntion, which should give us correct symbols for the path on emptyBoard
                                                    ##assuming that our Qfunction truly was the optimal one after the reinforcement learning sarsa
    """now that we madeWindyWorldAction, then
    we can calculate the difference in (row,col) between curState and newState, and 
    after that we just find out if we made diagonalmove (with windforce included), or straight arrowmove (windforce zero)
    
    IF both row,col are different compoonents compared to newState and curState, then make the diagonalmove appropriately
    ELSE make the straightmove appropriately
    
    This thing above should be one inside getWalkedPathSymbol()
    """

    curR = curState[0]
    curC = curState[1]
    emptyBoard[curR,curC] = sym

    curState = newState
    if gotStuckCounter > 200:
        raise Exception('optimalWalkPath across emptyBoard did not succeed (probably because Qfunc was not exactly optimal)')

print(emptyBoard)
kakka = 0 ##for debug only