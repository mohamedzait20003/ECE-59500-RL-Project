import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm



def reward_strategy(strategy, done, local_step, observation, action, next_observation, reward):
    if strategy == 'sparse':
        # give penalty for falling into the hole
        if done and next_observation != 15:
            reward = -1

        
    elif strategy == 'v1':
        # give penalty for staying in ground
        if reward == 0:
            reward = -0.001
            
        # give penalty for falling into the hole
        if done and next_observation != 15:
            reward = -1

        if local_step == 100:
            done = True #prevent infinite episode
            reward = -1

        if observation == next_observation: # prevent meaningless actions
            reward = -1
    else:
        raise NotImplementedError

    return reward



class Q_learning:
    def __init__(self, env, result_dir, potential_reward_shaping=False, strategy="sparse", gamma=0.8, alpha=0.1, eps=0.1, render=False, max_episode=1000):
        self.state_dim = env.observation_space.n
        self.action_dim = env.action_space.n
        
        self.env = env
        self.strategy = strategy
        self.result_dir = result_dir
        self.potential_reward_shaping = potential_reward_shaping
                
        self.nrow = int(env.observation_space.n**(0.5))
        self.ncol = int(env.observation_space.n**(0.5))
        
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.render = render
        self.max_episode = max_episode

        self.q = np.zeros([self.state_dim, self.action_dim])

    def action(self, s):
        if np.random.random() < self.eps:
            action = np.random.randint(low=0, high=self.action_dim - 1)
        else:
            action = np.argmax(self.q[s,:])

        return action

    def observation_to_map(self, observation):
        col = observation // self.nrow
        row = observation % self.nrow
        return row, col


    def potential_function(self, observation):
        row, col = self.observation_to_map(observation)
        goal_row, goal_col = self.observation_to_map(15)  # goal position
        # Using Manhattan distance as potential
        distance = abs(row - goal_row) + abs(col - goal_col)
        potential = -distance / (self.nrow + self.ncol)  # Normalized potential
        return potential
    
    def run(self):
        self.success = 0

        self.episode_reward = []
        self.episode_length = []

        for episode in tqdm(range(self.max_episode)):
            observation, _ = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            local_step = 0

            while not done and not truncated:

                action = self.action(observation)
                next_observation, reward, done, truncated, info  = self.env.step(action)
                
                if self.render:
                    self.env.render(title=f"Episode {episode} / step {local_step}", q=self.q)
                    os.makedirs(os.path.join(self.result_dir, 'renders'), exist_ok=True)
                    plt.savefig(f"{self.result_dir}/renders/render_episode_{episode}_step_{local_step}.png")
                    plt.close()

                
                reward = reward_strategy(self.strategy, done, local_step, observation, action, next_observation, reward)

                if self.potential_reward_shaping:
                    current_potential_reward = self.potential_function(observation)
                    next_potential_reward = self.potential_function(next_observation)
                    # q-learning with potential-based reward shaping
                    self.q[observation, action] = self.q[observation, action] + self.alpha*(reward + self.gamma*next_potential_reward - current_potential_reward + self.gamma*np.max(self.q[next_observation,:]) - self.q[observation, action])
                else:
                    # q-learning update
                    self.q[observation, action] = self.q[observation, action] + self.alpha*(reward + self.gamma*np.max(self.q[next_observation,:]) - self.q[observation, action])
                
                observation = next_observation
                episode_reward += reward
                local_step += 1

            #print("Episode: {}, Step: {}, Episode_reward: {}".format(episode, local_step, episode_reward))
            self.episode_reward.append(episode_reward)
            self.episode_length.append(local_step)

            if observation == 15:
                self.success += 1


        print("Training finished.\nSuccess rate: {:.2f}%".format(self.success/self.max_episode*100))

