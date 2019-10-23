import numpy
from copy import copy

class ModelOptimizer:
    def __init__(self, model):
        self.model = model
        self.optimizations = []
        self.models_to_average = 5
    
    def get_model_score(self):
        scores = []
        for i in range(self.models_to_average):
            print(f"Iteration: {i + 1}")
            self.model.reset()
            self.model.train(render=False)
            scores.append(self.model.score)
        return numpy.mean(scores)
    
    def optimize(self, optimization_params):
        scores = []
        param_configs = self.generate_model_params(optimization_params)
        for param_config in param_configs:
            for param_name in param_config:
                self.model.__dict__[param_name] = param_config[param_name]
            print(param_config)
            score = self.get_model_score()
            scores.append((score, param_config))
        return max(scores, key=lambda score: score[0])

    def generate_model_params(self, optimizationParams, existing_params={}):
        param_name, generator = optimizationParams[0]
        new_model_params = []
        for param_value in generator:
            params = copy(existing_params)
            params[param_name] = param_value
            if len(optimizationParams) != 1:
                new_model_params += self.generate_model_params(
                    optimizationParams[1:],
                    existing_params = params
                )
            else:
                new_model_params.append(params)
        return new_model_params
