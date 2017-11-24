#!/usr/bin/env python3

from game import Game
from rl_agent import RLAgent
from random_agent import Agent

import sys
from collections import deque

if __name__ == '__main__':
    agent = RLAgent()
    opponent = Agent(1) # random agent
    nb_games = 0
    history_length = 10000 # number of wins/illegals considered "recent"
    wins = deque()
    illegals = deque()


    try:
        while True:
            game = Game(5, 1, agent, opponent)
            while not game.over:
                game.turn()

            # count wins
            won = game.get_winning_team() == game.teams[0]
            wins.append(int(won))
            while len(wins) > history_length:
                wins.popleft()

            # count losses due to illegal actions
            illegal = agent.end_match()
            illegals.append(int(illegal))
            while len(illegals) > history_length:
                illegals.popleft()

            nb_games += 1
            agent.reset_game()
            opponent.reset_game()

            if nb_games % 100 == 0:
                sys.stdout.write('\rgames: %i, recent win rate: %.2f%% illegal rate: %.2f%%  ' % (nb_games, 100. * sum(wins) / len(wins), 100. * sum(illegals) / len(illegals)))
                sys.stdout.flush()

    except KeyboardInterrupt:
       agent.shutdown()
       sys.stdout.write('\n')
       print('Trained %i games.' % nb_games)
