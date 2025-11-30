import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

class NewRender(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def render(self, title='Environment', q=None):
       
        fig = plt.figure(figsize=(8, 4))

        ax_list = [fig.add_subplot(1, 2, 1)]
        img = ax_list[0].imshow(self.env.render())  # prepare to render the environment by using matplotlib and ipython display
        ax_list[0].set_title(title)
        ax_list.append(fig.add_subplot(1, 2, 2))

        ax_index = 1
        nrow, ncol = int(self.env.observation_space.n**(0.5)), int(self.env.observation_space.n**(0.5))
        visualize_q(q, ax_list[ax_index], nrow, ncol)

        for ax in ax_list:
            ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

        plt.tight_layout()
        
    

def visualize_model(model, ax, nrow=4, ncol=4, title='Model') -> None:
    # visualize model
    visualize_q(model, ax, nrow, ncol, title)


def visualize_q(q_func, ax, nrow=4, ncol=4, title='State-Action Value') -> None:
    # visualize state-action value

    q_func = q_func.reshape([nrow, ncol, 4])

    h_ticks = np.arange(-0.5, nrow + 0.5, 1)
    v_ticks = np.arange(-0.5, ncol + 0.5, 1)

    # plot the state-action function

    im = ax.imshow(np.zeros([nrow, ncol]), alpha=0.)

    for tick in h_ticks:
        ax.axhline(tick, color='black', linewidth=0.3)

    for tick in v_ticks:
        ax.axvline(tick, color='black', linewidth=0.3)

    for h in range(len(h_ticks) - 1):
        for v in range(len(v_ticks) - 1):
            ax.plot([v_ticks[v], v_ticks[v + 1]], [h_ticks[h], h_ticks[h + 1]], color='black', linewidth=0.3)
            ax.plot([v_ticks[v], v_ticks[v + 1]], [h_ticks[h + 1], h_ticks[h]], color='black', linewidth=0.3)

    for (j, i, k), q in np.ndenumerate(q_func):
        if k == 0:
            ax.text(i - 0.3, j, np.round(q, 2), ha='center', va='center', fontsize=8)  # left
        elif k == 1:
            ax.text(i, j + 0.3, np.round(q, 2), ha='center', va='center', fontsize=8)  # bottom
        elif k == 2:
            ax.text(i + 0.3, j, np.round(q, 2), ha='center', va='center', fontsize=8)  # right
        elif k == 3:
            ax.text(i, j - 0.3, np.round(q, 2), ha='center', va='center', fontsize=8)  # top
        else:
            raise ValueError
    ax.set_title(title)


def visualize_v(v_func, ax, nrow=4, ncol=4, title='State Value') -> None:
    # visualize state value

    v_func = v_func.reshape([nrow, ncol])

    h_ticks = np.arange(-0.5, nrow + 0.5, 1)
    v_ticks = np.arange(-0.5, ncol + 0.5, 1)

    # plot the state-value function

    im = ax.imshow(np.zeros([nrow, ncol]), alpha=0.)

    for tick in h_ticks:
        ax.axhline(tick, color='black', linewidth=0.3)

    for tick in v_ticks:
        ax.axvline(tick, color='black', linewidth=0.3)

    for (j, i), value in np.ndenumerate(v_func):
        ax.text(i, j, np.round(value, 3), ha='center', va='center', fontsize=14)
        
    ax.set_title(title)


def visualize_policy(policy, ax, nrow=4, ncol=4, title='Policy') -> None:
    # visualize deterministic policy
    if len(policy.shape) == 2:
        visualize_q(policy, ax, nrow, ncol, title='Stochastic Policy')
    else:
        arrow = {0: '←', 1: '↓', 2: '→', 3: '↑'}

        policy = policy.reshape([nrow, ncol])

        h_ticks = np.arange(-0.5, nrow + 0.5, 1)
        v_ticks = np.arange(-0.5, ncol + 0.5, 1)

        im = ax.imshow(np.zeros([nrow, ncol]), alpha=0.)

        for tick in h_ticks:
            ax.axhline(tick, color='black', linewidth=0.3)

        for tick in v_ticks:
            ax.axvline(tick, color='black', linewidth=0.3)

        for (j, i), p in np.ndenumerate(policy):
            ax.text(i, j, arrow[p], ha='center', va='center', fontsize=36)

        ax.set_title(title)