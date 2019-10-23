import gym
import numpy
from progress.bar import Bar

class BaseGymEnv:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
    
    def choose_action(self):
        pass

    def update_model(self):
        pass
    
    def reset(self):
        self.baseline = None
        self.score = None

    def run(self, iterations=10000, **kwargs):
        self.get_baseline(**kwargs)
        self.train(iterations, **kwargs)
        self.score_model(**kwargs)
        self.print_score()

    def play_episodes(self, title="", iterations=0, **kwargs):
        bar = Bar(f"{title}: " + " " * (30 - len(title)), max=iterations)
        rewards = []
        for i in range(iterations):
            reward = self.play_episode(**kwargs)
            rewards.append(reward)
            bar.next()
        bar.finish()
        return rewards
    
    def train(self, iterations = 10000, **kwargs):
        rewards = self.play_episodes(
            title="Training", 
            iterations=iterations,
            train_model=True,
            **kwargs
        )
        self.score = self.get_score(rewards)
        return self
    
    def get_baseline(self, iterations=1000, **kwargs):
        choose_action_orig = self.choose_action
        self.choose_action = lambda **kwargs : self.env.action_space.sample()
        rewards = self.play_episodes(
            title="Finding Baseline",
            iterations=iterations,
            train_model=False,
            **kwargs
        )
        self.choose_action = choose_action_orig
        self.baseline = self.get_score(rewards)
        return self.baseline

    def get_score(self, rewards):
        return numpy.mean(rewards)
    
    def score_model(self, iterations=1000, **kwargs):
        rewards = self.play_episodes(
            title="Scoring Model", 
            iterations=iterations,
            train_model=False,
            **kwargs
        )
        self.score = self.get_score(rewards)
        return self.score

    def play_episode(
        self,
        render=True,
        step_by_step=False,
        train_model=True
    ):
        observation = self.env.reset()
        done=False
        reward=0

        while not done:
            if render:
                self.env.render()
            action = self.choose_action(observation=observation)
            new_observation, new_reward, done, _ = self.env.step(action)
            if train_model:
                self.update_model(
                    old_observation=observation,
                    new_observation=new_observation,
                    reward=new_reward,
                    action=action,
                    done=done
                )
            if step_by_step:
                input("Next [Enter]")

            reward += new_reward
            observation = new_observation
        return reward

    def print_score(self):
        print (f"""
------- Average Reward --------
    Baseline: {self.baseline}
    Model: {self.score}
-------------------------------
""")
