import numpy as np
import numpy.linalg as LA
import random
from datetime import datetime
from numpy import random as npRAND

random.seed(datetime.now())


rows_count = 4
columns_count = 4

def isTerminal(r, c):  # helper function to check if terminal state or regular state
    global rows_count, columns_count
    if r == 0 and c == 0:  # im a bit too lazy to check otherwise the iteration boundaries
        return True  # so that this helper function is a quick way to exclude computations
    if r == rows_count - 1 and c == columns_count - 1:
        return True
    return False



"""STARTING VARIABLES AND PARAMETERS"""
reward = -1
maxiters = 10000
alpha = 0.01
V = np.zeros((rows_count, columns_count))
actDict={0:"U",1:"R",2:"D",3:"L"}
epsilon = 0.4
constEpsilon = epsilon ## original epsilon was
actions = ["U", "R", "D", "L"]
returnsDict={}
QDict={}
actDict={0:"U",1:"R",2:"D",3:"L"}
policies = {}
probDist = [0.25, 0.25, 0.25, 0.25] ## up,right,down,left


"""printedpolicy is for printing at the end to check if algorithm
was optimal or how close to it, it was or wasnt
(you check if by looping thru the policy and choosign argmax action always)"""
printedPolicy = np.array([ ['T','A','A','A'],
                     ['A','A','A','A'],
                     ['A','A','A','A'],
                     ['A','A','A','T'] ])

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
                QDict[ ((r,c), act) ] = -9999  ## Maintain Q function value for each state-action pair
            else:
                QDict[  ((r,c), act) ] = 0




"""parameter state tuple
returns the chosenACtion from epsilonGreedy policy (string action_0)"""
def getActionFromEpsGreedyPolicy(S_t):
    global policies, actions
    curActDistr = policies[ S_t ].copy()
    chosenAct = npRAND.choice( actions, 1, p=curActDistr )
    chosenAct = str(chosenAct[0])
    return chosenAct


"""parameters state tuple, currentArgmaxAction
void return,
updates global policies action distribution"""
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




def getRandomStartState():
    illegalState = True
    while illegalState:
        r = random.randint(0, 3)
        c = random.randint(0, 3)
        if (r == 0 and c == 0) or (r == 3 and c == 3):
            illegalState = True
        else:
            illegalState = False
    return r, c


def getState(row, col):
    if row == -1:
        row = 0
    elif row == 4:
        row = 3
    if col == -1:
        col = 0
    elif col == 4:
        col = 3
    return row, col


def getRandomAction():
    global actDict
    return actDict[random.randint(0, 3)]


def getMeanFromReturns(oldMean, n, curVal):
    if n == 0:
        raise Exception('Exception, incrementalMeanFunc, n should not be less than 1')
    elif n == 1:
        return curVal
    elif n >= 2:
        newMean = (float) ( oldMean + (1.0 / n) * (curVal - oldMean) )
        return newMean



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



"""get the best action 
returns string action
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




print("hello TD SARSA, oldGridWorld!")


for iteration in range(1, maxiters+1):
    print("TD_SARSA_iterCount == ", iteration, "eps == ", epsilon,"\n")

    if iteration % 20 == 0: ## get random seed periodically to improve randomness performance in episodeGeneration
        random.seed(datetime.now())
    if iteration % 50 == 0:
        epsilon = epsilon * 0.975  ## reduce epsilon, favor the exploration near start, and favor exploitation near end

    for row in range(4):
        for col in range(4):
            if not isTerminal(row,col):
                """episode begins in earnest! now"""
                S = (row, col)  ## get startState and startAction
                A = getActionFromEpsGreedyPolicy(S)
                optA = getArgmaxActQ(S)  ## NOTE!!! important! we must explicitly get argmaxAct, because getActFromEpsGreedy is not guaranteed to return optAct!!!
                updateEpsGreedyPolicy(S, optA)  ## update eps-greedy policy with current (S,Aopt) probabilities,
                while not isTerminal(S[0], S[1]):
                    r0 = S[0]   ## simply decompose curState into components
                    c0 = S[1]
                    Sprime, R, endedInTerminalState = makeAction(S, A)
                    Aprime = getActionFromEpsGreedyPolicy(Sprime)
                    r1 = Sprime[0]    ## simple decompose newState into components
                    c1 = Sprime[1]
                    curValue = QDict[ S, A ]
                    newValue = QDict[ Sprime, Aprime ]
                    QDict[ (S, A) ] = (curValue + alpha * ( R + 1.0 * newValue- curValue ))
                    S = Sprime
                    A = Aprime
                    optAprime = getArgmaxActQ(Sprime) ## NOTE!!! important! we must explicitly get argmaxAct, because getActFromEpsGreedy is not guaranteed to return optAct!!!
                    updateEpsGreedyPolicy( Sprime, optAprime ) ## update eps-greedy policy with new (Sprime,AprimeOpt)probabilities

print("TD_Sarsa Qfunc0 was \n\n", QDict)



"""print the epsGreedyPolicy"""
print("\n the eps greedy optimal policy was \n\n")
for SApair in QDict:
    if (SApair[0] != (0,0)) and (SApair[0] != (3,3)):
        curState = SApair[0]
        optA = getArgmaxActQ( curState )
        r0 = curState[0]
        c0 = curState[1]
        printedPolicy[r0,c0] = optA
print(printedPolicy)