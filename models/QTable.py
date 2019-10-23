import numpy
import random
from progress.bar import Bar
from models.BaseGymEnv import BaseGymEnv

class QTable(BaseGymEnv):
    def __init__(self, env_name, discount_rate=.5, learning_rate=.5, noise=1):
        super().__init__(env_name)
        self.create_table()
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.noise = noise
    
    def create_table(self, initial_val=0):
        possibleObservations = self.env.observation_space.n
        possibleActions = self.env.action_space.n
        QTable = numpy.empty((possibleObservations, possibleActions), dtype=float)
        QTable.fill(initial_val)
        self.QTable = QTable
        self.episodes_played = 0
        return self
    
    def choose_action(self, observation=None):
        actions = self.QTable[observation]
        self.episodes_played += 1
        most_reward_possible = max(actions)
        threshold = self.noise / self.episodes_played
        actions_to_choose = [ (action, reward) for (action, reward) in enumerate(actions) if reward >= (most_reward_possible - threshold)]
        return random.choice(actions_to_choose)[0]

    def update_model(self, old_observation=None, action=None, new_observation=None, reward=None, done=None):
        possible_future_reward = max(self.QTable[new_observation])
        self.QTable[old_observation, action] = ( \
            (1 - self.learning_rate) * self.QTable[old_observation, action] + \
            (self.learning_rate) * (reward + possible_future_reward * self.discount_rate) \
        )
    
    def reset(self):
        super().reset()
        self.create_table()
