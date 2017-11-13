#!/usr/bin/env python3

from game import Game
from rl_agent import RLAgent
from random_agent import Agent

import sys

if __name__ == '__main__':
    agent = RLAgent()
    opponent = Agent(1) # random agent
    nb_games = 0
    wins = []

    try:
        while True:
            game = Game(5, 1, agent, opponent)
            while not game.over:
                game.turn()

            won = game.get_winning_team() == game.teams[0]
            if won:
                wins.append(1)
            else:
                wins.append(0)
            while len(wins) > 1000:
                del wins[0]

            if nb_games % 10 == 0:
                sys.stdout.write('\rgames: %i, recent win rate: %2.f%%   ' % (nb_games, 100. * sum(wins) / len(wins)))
                sys.stdout.flush()

            if nb_games % 3000 == 0:
                print('\nW1:')
                print(agent.W1)
                print('W2:')
                print(agent.W2)

            agent.end_match(won)
            nb_games += 1
            agent.reset_game()
            opponent.reset_game()
    except KeyboardInterrupt:
       agent.shutdown()
       sys.stdout.write('\n')
       print('Trained %i games.' % nb_games)
