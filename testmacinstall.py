"""
Sample script for the Arcade Learning Environment's Python interface

Usage:
    demo.py <rom_file> [options]

Options:
    --iters=N    Number of iterations to run [default: 5]
    --display    Display the game being played. Uses SDL.

@author: Alvin Wan
@site: alvinwan.com
"""

import docopt
import random
import pygame
import sys

from ale_python_interface import ALEInterface


def main():
    arguments = docopt.docopt(__doc__, version='ALE Demo Version 1.0')

    pygame.init()

    ale = ALEInterface()
    ale.setInt(b'random_seed', 123)
    ale.setBool(b'display_screen', True)
    ale.loadROM(str.encode(arguments['<rom_file>']))

    legal_actions = ale.getLegalActionSet()

    rewards, num_episodes = [], int(arguments['--iters'] or 5)
    for episode in range(num_episodes):
        total_reward = 0
        while not ale.game_over():
            total_reward += ale.act(random.choice(legal_actions))
        print('Episode %d reward %d.' % (episode, total_reward))
        rewards.append(total_reward)
        ale.reset_game()

    average = sum(rewards)/len(rewards)
    print('Average for %d episodes: %d' % (num_episodes, average))

if __name__ == '__main__':
    main()

