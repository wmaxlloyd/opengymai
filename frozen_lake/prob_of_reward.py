import gym
import numpy
import random

env = gym.make("FrozenLake-v0")

# Initialize Constants
numEpisodes = 1000
rewardsWon = []
winCount = 0
initialReward = 0

# Initialize QTable
lakeTiles = env.observation_space.n
numberOfActions = env.action_space.n
initialProb = 1 / numberOfActions
PTable = numpy.empty((lakeTiles, numberOfActions), dtype=float)
PTable.fill(initialProb)

# Learning Weights
discountRate = .95 # Between 0 and 1, this is the weight at which we want to determine the importance of future decisions
learningRate = .8 # Not sure what this is yet...


def episode(episodeNum, step=False, absolute=False):
    print(f"Starting Episode: {episodeNum}")
    location = env.reset()
    done = False
    reward = initialReward

    while not done:
        action = chooseAction(PTable[location])
        if absolute:
            action = numpy.argmax(PTable[location])
        newLocation, newReward, done, _ = env.step(action)
        if step:
            renderEnv(env, "Final")
            print (PTable[location], action)
            input("Next:")
            
        bestFutureProb = numpy.max(PTable[newLocation])
        reinforcement = 1
        if newReward:
            reinforcement = (1 / learningRate)
        if done and not newReward:
            reinforcement = learningRate

        PTable[location, action] *= reinforcement
        PTable[location] = [ actionProb / numpy.sum(PTable[location]) for actionProb in PTable[location]]

        location = newLocation
        reward += newReward

    return reward

def chooseAction(actionsProbs):
    actionTrigger = random.random()
    for (action, actionProb) in enumerate(actionsProbs):
        actionTrigger -= actionProb
        if actionTrigger <= 0:
            return action

# Run Scenario
print("Training")
for i in range(numEpisodes):
    reward = episode(i)
    rewardsWon.append(reward) 
print("Done!")

tests = 0
wins = 0
while True:
    tests += 1
    reward = episode("Final", step=True, absolute=True)
    if reward:
        wins += 1
    print(wins, tests)
    input("Final:")


