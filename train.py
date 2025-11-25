import importlib, argparse, yaml, os, datetime
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from src.utils import NewRender

def parse_args(parser):
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args = parse_args(args)
    config = yaml.safe_load(open(args.config, 'r'))

    # Create save directory if it doesn't exist
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = os.path.join('results', config['algorithm'], date)
    os.makedirs(result_dir, exist_ok=True)

    # Add result_dir to config
    config['result_dir'] = result_dir

    # Copy config file to the result directory for future reference
    with open(os.path.join(result_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)


    env = gym.make(config['environment']['name'], render_mode='rgb_array', is_slippery=config['environment']['is_slippery'], map_name=config['environment']['map_name'])#define the environment.
    env = NewRender(env)

    trainer = importlib.import_module(f"src.algorithms").__dict__[config['algorithm']](env, config['result_dir'], **config['hyperparameters'])
    trainer.run()

    # Plot episode reward over time
    plt.plot(trainer.episode_reward, label='Episode Reward')
    plt.plot(np.convolve(trainer.episode_reward, np.ones(10)/10, mode='valid'), label='Smoothed') # smooth curve
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Episode Reward over Time')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'episode_reward.png'))