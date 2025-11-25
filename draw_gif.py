import imageio, argparse
import os, glob


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./results/sparse/2025-11-26_04-26-50/renders', help='Directory containing input images.')
    parser.add_argument('--episode', type=int, default=0, help='Episode number to create GIF for. -1 for all episodes.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parser_args()
    input_dir = args.input_dir
    episode = args.episode

    if episode == -1:
        image_files = sorted(glob.glob(os.path.join(input_dir, 'render_episode_*')))
    else:
        image_files = sorted(glob.glob(os.path.join(input_dir, f'render_episode_{episode}_step_*.png')))

    images = []
    for filename in image_files:
        images.append(imageio.v2.imread(filename))
    output_gif = os.path.join(input_dir.replace('/renders', ''), f'episode_{episode}_animation.mp4')
    imageio.mimsave(output_gif, images, fps=5)
    print(f"GIF saved to {output_gif}")