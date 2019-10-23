import gym
import numpy

env = gym.make("FrozenLake-v0")

# Initialize Constants
initialReward = 0
num_episodes = 10000
rewardsWon = []
winCount = 0

# Initialize QTable
lakeTiles = env.observation_space.n
numberOfActions = env.action_space.n
QTable = numpy.empty((lakeTiles, numberOfActions), dtype=float)
QTable.fill(initialReward)

# Learning Weights
discountRate = .95 # Between 0 and 1, this is the weight at which we want to determine the importance of future decisions
learningRate = .1 # Not sure what this is yet...


def episode(episode_num):
    print(f"Starting Episode: {episode_num}")
    location = env.reset()
    done = False
    reward = initialReward

    while not done:
        renderEnv(env, episode_num)
        actionsWithBestOutcome = getActionsWithBestOutcome(
            QTable[location],
            episode_num
        )
        action = numpy.random.choice(actionsWithBestOutcome)
        new_location, new_reward, done, _ = env.step(action)
        # Calculating value of Q Table
        QTable[location, action] =  ( \
        QTable[location, action] * (1 - learningRate) # Cuts current reward according to learning rate \
        + learningRate * ( # Determines what reward should be added based on current and future rewards \ 
            new_reward \
            + discountRate * numpy.max(QTable[new_location]) # Calculates max future reward but discounts it according to discout rate \
        ))
        if new_reward:
            global winCount
            winCount += 1
        location = new_location
        reward += new_reward

    renderEnv(env, episode_num)
    return reward

def getActionsWithBestOutcome(actions, episode_num):
    maxReward = numpy.max(actions)
    allowedVariance = discountRate ** episode_num
    allowedLowestReward = (1 - allowedVariance) * maxReward
    possibleActions = []
    for (action, reward) in enumerate(actions):
        if reward >= allowedLowestReward:
            possibleActions.append(action)
    return possibleActions

def renderEnv(env, episode_num):
    print(chr(27) + "[2J")
    print(f"Episode: {episode_num}. Wins: {winCount}")
    env.render()
# Run Scenario
for i in range(num_episodes):
    reward = episode(i)
    rewardsWon.append(reward) 
print (QTable)