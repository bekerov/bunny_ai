#James Marcus Hughes
#Last edit: October 25, 2016


import sys

#All possible states in the map
S = [(1,2), (1,3), (2,1), (2,2), (2,3), (2,4), (3,1), (3,2), (3,3), (3,4)]

'''
   Gives reward based upon current state, 
   base_reward is anywhere not specially defined
'''
def R(state, base_reward):
    #in waterfall
    if state == (1,2) or state == (1,3):
        return -100
    #at grass feeding
    elif state == (2,4):
        return 100
    else:
        return base_reward

'''Declares if a state, s, is a terminal state'''
def terminal(s):
    return s == (1,2) or s == (1,3) or s == (2,4)

'''All legal actions given a current state, s'''
def A_fn(s):
    if terminal(s):
        return []
    #Can move in any direction provided it goes into a legal state
    moves = [(0,1), (0,-1), (1,0), (-1,0)]
    valid = []
    for move in moves:
        result = (s[0] + move[0], s[1] + move[1])
        #moves into legal state
        if result in S:
            valid.append(move)
    return valid

'''Describes the probability to go from state s to sprime given action a'''
def P(s, sprime, a):
    if a not in A_fn(s):
        return 0
    #assert a in A_fn(s)
    grass = {(2,1), (3,1), (2,4), (3,4)}
    if s in grass:
        if sprime == (s[0] + a[0], s[1] + a[1]):
            return 1
        else:
            return 0
    else: #in the water
        if a == (1,0) or a == (0,1) or a == (0,-1):
            #drifting
            if sprime == (s[0] - 1, s[1]):
                return 0.25
            #not drifting and got to target
            elif sprime == (s[0] + a[0], s[1] + a[1]): 
                return 0.75
            #neither pulled by current or reached target
            else: 
                return 0
        else: #a == (-1,0), trying to go left
            return 1 if sprime == (s[0] + a[0], s[1] + a[1]) else 0

'''Describes all legal resulting states from state s'''
def sprimes(s):
    legal_actions = A_fn(s)
    sprime_states = []
    for a in legal_actions:
        sprime_states.append((s[0] + a[0], s[1] + a[1]))
    return sprime_states

'''
    Performs value iteration to determine optimal utilities.
    Will terminate when change in utilites is less than epsilon
    Can force to terminate after n trials by setting imax=n
    Here S is all legal states and R is the reward function
'''
def value_iteration(S, R, epsilon=1E-5, gamma=1.0, base_reward=-5, imax=None):
    #set up necessary local variables
    delta = 0.0
    U = {}
    Uprime = {}
    i = 0
    #begin with utilites of zero
    for s in S:
        U[s] = 0
        Uprime[s] = 0
        
    
    while True:
        #U[s] <- U'[s] for all s in S
        #U = Uprime, except done by element to avoid memory overwrite
        for s in S:
            U[s] = Uprime[s]
        
        delta = 0
        for s in S:
            #a terminal state with no legal actions
            if terminal(s): 
                Uprime[s] = R(s, base_reward)
            else:
                #Uses the Bellman equation
                summation = [sum([P(s, sprime, a) * U[sprime] for sprime in S]) for a in A_fn(s)]
                Uprime[s] = R(s, base_reward) + gamma * max(summation)
            #this state has a greater change than already seen
            if abs(Uprime[s] - U[s]) > delta:
                delta = abs(Uprime[s] - U[s])
        i+=1
        #biggest change is smaller than threshold quit
        if delta <= epsilon * (1-gamma) / gamma: 
            break
        #if maximum iterations are set, quit
        if imax != None and i >= imax:
            break
    return U

if __name__ == "__main__":
    print("*"*80)
    print("*" * 9, "Bunny world AI for CS 373, Williams College by Marcus Hughes", "*" * 9)
    print("*"*80)

    print()
    print("Please provide an argument variable upon calling as the the base reward for")
    print("other squares than the terminal states e.g. python cs373_hw3.py -5 for a")
    print("reward of -5 in all the other squares. The program will use value iteration")
    print("to find an optimal utility and policy. For more explanation, see the handout")
    print("and the solutions pdf associated in this zipped file. HAPPY HOPPING!!!")
    print()
    base_reward = float(sys.argv[1])
    U = value_iteration(S, R, base_reward = base_reward)
    print()
    print("Optimal Utilities are :")
    print("state   :  utility")
    for s in S:
        print(s, " : ", U[s])

    print()
    print()
    print("Optimal Policy :")
    print("state   :  set of optimal actions with estimate of ____ ")
    for s in S:
        best_actions = []
        best_action_utility = -10**10
        results = [(a, sum([ P(s, sprime, a) * U[sprime] for sprime in sprimes(s)])) for a in A_fn(s)]
        for a, u in results:
            if u > best_action_utility and u != 0:
                best_action_utility = u
                best_actions = [a]
            elif u == best_action_utility and u!= 0:
                best_actions.append(a) 
        if not terminal(s):
            print(s," : ", best_actions, " with estimate of ", best_action_utility)
    print()
