import sys
sys.path.append("/Users/mlloyd/Personal_Code/gym_ai")

from models.QTable import QTable
from models.OptimizeModel import ModelOptimizer

model = QTable(
    "FrozenLake-v0",
    # discountRate=.9, # Between 0 and 1, this is the weight at which we want to determine the importance of future decisions
    # learningRate=.5, # Not sure what this is yet...
    # noise=20
)

optimized_params = ModelOptimizer(model).optimize(
    [
        ("discount_rate", [0, .1,.2, .3, .4, .5, .6, .7, .8, .9, 1]),
        ("learning_rate", [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]),
        ("noise", [0, 5, 10, 25, 50, 100, 200, 500, 10000])
    ]
)
# (0.66882, {'discount_rate': 0.9, 'learning_rate': 0.8, 'noise': 5})
print(optimized_params)