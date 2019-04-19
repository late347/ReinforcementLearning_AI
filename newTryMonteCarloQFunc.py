
import numpy as np
import numpy.linalg as LA
import random
from datetime import datetime

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



"""NOTE about maxiters!!!
the Monte-Carlo every visit algorithm implements total amount of iterations with formula
totalIters = maxiters * nonTerminalStates * possibleActions
totalIters = 5000 * 14 * 4
totalIters = 280000

in other words, there will be 5k iterations per each state-action pair
in other words there will be an early exploration phase where policy willnot be updated,
but the gridworld will be explored with randomwalk policy, gathering Qfunc information, 
and returnDict information.

in early phase there will be about 27 iterations for each state-action pair during,
non-policy-updating exploration 
(maxiters * explorationFactor) / (stateACtionPairs) = 7500 *0.2 /56

after that early exploring with randomwalk,
then we act greedily w.r.t. the Q-function, 
for the rest of the iterations to get the optimal deterministic policy
"""
maxiters = 7500
explorationFactor = 0.2 ## explore that percentage of the first maxiters rounds, try to increase it, if you get stuck in foreverloop, in QEpisode function
reward = -1
actions = ["U", "R", "D", "L"]
V = np.zeros((rows_count, columns_count))
returnsDict={}
QDict={}
actDict={0:"U",1:"R",2:"D",3:"L"}
policies = np.array([ ['T','A','A','A'],
                     ['A','A','A','A'],
                     ['A','A','A','A'],
                     ['A','A','A','T'] ])





"""returnsDict, for each state-action pair, maintain (mean,visitedCount)"""
for r in range(rows_count):
    for c in range(columns_count):
        if not isTerminal(r, c):
            for act in actions:
                returnsDict[ ((r, c), act) ] = [0, 0] ## Maintain Mean, and VisitedCount for each state-action pair



""" Qfunc, we maintain the action-value for each state-action pair"""
for r in range(rows_count):
    for c in range(columns_count):
        if not isTerminal(r, c):
            for act in actions:
                QDict[ ((r,c), act) ] = -9999  ## Maintain Q function value for each state-action pair






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
        if (r == 0 and c == 0) or (r == 3 and c == 3):
            illegalState = True
        else:
            illegalState = False
    return r, c

def getState(row, col):
    if row == -1:
        row = 0  # helper func for the exercise:1
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
    newMean = 0
    if n == 0:
        raise Exception('Exception, incrementalMeanFunc, n should not be less than 1\n')
    elif n == 1:
        return curVal
    elif n >= 2:
        newMean = (float) ( oldMean + (1.0 / n) * (curVal - oldMean) )
        return newMean


"""get the best action 
returns string action
parameter is state tuple (r,c)"""
def getArgmaxActQ(S_t):
    global QDict
    qvalList = []
    saList = []

    """for example get together
    s1a1, s1a2, s1a3, s1a4
    find which is the maxValue, and get the action which caused it"""
    sa1 = (S_t, "U")
    sa2 = (S_t, "R")
    sa3 = (S_t, "D")
    sa4 = (S_t, "L")
    saList.append(sa1)
    saList.append(sa2)
    saList.append(sa3)
    saList.append(sa4)

    q1 = QDict[sa1]
    q2 = QDict[sa2]
    q3 = QDict[sa3]
    q4 = QDict[sa4]
    qvalList.append(q1)
    qvalList.append(q2)
    qvalList.append(q3)
    qvalList.append(q4)

    maxQ = max(qvalList)
    ind_maxQ = qvalList.index(maxQ)  # gets the maxQ value and the index which caused it

    """when we have index of maxQval, then we know which sa-pair
    gave that maxQval => we can access that action from the correct sa-pair"""
    argmaxAct = saList[ind_maxQ][1]
    return argmaxAct




"""QEpisode generation func
returns episodeList
parameters are starting state, starting action"""
def QEpisode(r, c, act):

    """ideally, we should not get stuck in the gridworld...but,
    but sometiems when policy transitions from the first episode's policy == randomwalk,
    then, on second episode sometimes we get stuck in foreverloop in episode generation
    usually the only choice then seems to restart the entire policy into randomwalk ??? """

    global reward
    global policies

    """NOTE! r,c will both be local variables inside this func
    they denote the nextState (s') in this func"""
    stepsTaken = 0
    curR = r
    curC = c
    episodeList = [ ((r, c), act, reward) ]  # add the starting (s,a) immediately

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

    while not stateWasTerm:
        if policies[curR, curC] == "A":
            act = getRandomAction()  ## """get the random action from policy"""
        else:
            act = policies[curR, curC]  ## """get the deterministic action from policy"""

        if act == "U":  ## up
            r -= 1
        elif act == "R":  ## right
            c += 1
        elif act == "D":  ## down
            r += 1
        else:  ## left
            c -= 1
        stepsTaken += 1

        r, c = getState(r, c)
        stateWasTerm = isTerminal(r, c)
        episodeList.append( ((curR, curC), act, reward) )
        if not stateWasTerm:
            curR = r
            curC = c
        if stepsTaken >= 100000:
            raise Exception("Exception raised, because program got stuck in MC Qepisode generation...\n")


    return episodeList




print("montecarlo program starting...\n")
""" MOnte Carlo Q-function, exploring starts, every-visit, estimating Pi ~~ Pi* """

"""It appears that the Qfunction apparently can be unreliable in the early episodes rounds, so we can avoid getting 
stuck in foreverloop because of unreliable early episodes, BUT...

we gotta delay updating the policy, until we have explored enough for a little bit...
so our Qfunction has reliable info inside of it, to base the decision on, later..."""
Q_function_is_reliable = False ## variable shows if we are currently updating the policy, or just improving Q-function and exploring


for iteration in range(1, maxiters+1): ## for all episodes

    print("curIter == ", iteration, ", QfunctionIsReliable == ", Q_function_is_reliable )
    print("\n")
    if iteration % 20 == 0: ## get random seed periodically to improve randomness performance
        random.seed(datetime.now())

    for r in range(4):  ## for every non-terminal-state
        for c in range(4):
            if not isTerminal(r,c):
                startR = r
                startC = c
                for act in actions: ## for every action possible
                    startAct = act
                    sequence = QEpisode(startR, startC, startAct)  ## generate Q-sequence following policy Pi, until terminal-state (excluding terminal)
                    G = 0

                    for t in reversed(range(len(sequence))): ## iterate through the timesteps in reversed order
                        S_t = sequence[t][0] ## use temp variables as helpers
                        A_t = sequence[t][1]
                        R_t = sequence[t][2]
                        G += R_t ## increment G with reward, gamma discount factor is zero
                        visitedCount = returnsDict[S_t, A_t][1]
                        visitedCount += 1

                       ## if (S_t, A_t, -1) not in sequence[:t]: ## This is how you COULD have done the first-visit MC, but we do every-visit now...
                        if visitedCount == 1: ## special case in iterative mean algorithm, the first visit to any state-action pair
                            curMean = 9999
                            curMean = getMeanFromReturns(curMean, visitedCount, G)
                            returnsDict[S_t, A_t][0] = curMean ## update mean
                            returnsDict[S_t, A_t][1] = visitedCount ## update visitedcount
                        else:
                            curMean = returnsDict[S_t, A_t][0] ## get temp mean from returnsDict
                            curMean = getMeanFromReturns(curMean, visitedCount, G) ## get the new temp mean iteratively
                            returnsDict[S_t, A_t][1] = visitedCount ## update visitedcount
                            returnsDict[S_t, A_t][0] = curMean ## update mean


                        QDict[S_t, A_t] = returnsDict[S_t, A_t][0] ## update the Qfunction with the new mean value
                        tempR = S_t[0] ## temp variables simply to disassemble the tuple into row,col
                        tempC = S_t[1]

                        if iteration >= round(maxiters * explorationFactor): ## ONLY START UPDATING POLICY when we have reliable estimates for Qfunction, that is when iteration > maxiter/10
                            Q_function_is_reliable = True
                            policies[tempR, tempC] = getArgmaxActQ(S_t) ## update policy based on argmax_a[Q(S_t)]


print("optimal policy with Monte-Carlo, every visit was \n")
print("\n")
print(policies)