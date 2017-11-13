import sys
import math
import random
import time
from collections import Counter
import numpy as np


class Player:
    '''stores the position of a figure and whether we can control it'''
    def __init__(self, friendly, position=None):
        self.friendly = friendly
        self.position = None
        if position is not None:
            self.set_position(position)

    def set_position(self, position):
        self.position = list(position)


class Agent:
    '''main AI unit, storing the world and making decisions'''
    def __init__(self, nb_players_per_side):
        self.nb_players_per_side = nb_players_per_side
        self.reset_game()

    def reset_game(self):
        '''sets the agent up for a new game'''
        self.world = None
        self.players = [Player(True) for _ in range(self.nb_players_per_side)] + [Player(False) for _ in range(self.nb_players_per_side)]

    def set_world(self, world, scores):
        to_int = lambda c: 4 if c == '.' else int(c)
        self.world = [list(map(to_int, line)) for line in world]

    def iter_neighbors(self, position):
        '''yields reachable neighbors of a cell'''
        for x in range(position[0] - 1, position[0] + 2):
            for y in range(position[1] - 1, position[1] + 2):
                # self
                if x == position[0] and y == position[1]:
                    continue
                # o.o.b.
                if x < 0 or y < 0 or x >= len(self.world) or y >= len(self.world):
                    continue
                # unplayable
                if self.world[y][x] == 4:
                    continue
                yield x, y

    def get_cell_player_idx(self, position):
        '''returns the player index if there is a player blocking the cell, -1 otherwise'''
        for i, player in enumerate(self.players):
            if player.position == list(position):
                return i
        return -1

    def iter_legal_actions(self, player_idx):
        '''yields legal actions as tuples (action_type, player_idx, pos1, pos2)'''
        x, y = self.players[player_idx].position
        if x == -1 and y == -1:
            return

        # move
        for nx, ny in self.iter_neighbors([x, y]):
            # occupied?
            other_idx = self.get_cell_player_idx([nx, ny])
            if other_idx >= 0 and not self.players[other_idx].friendly:
                # enemy -> push
                for nnx, nny in self.iter_neighbors([nx, ny]):
                    # illegal push direction
                    if abs((nx - x) - (nnx - nx)) + abs((ny - y) - (nny - ny)) > 1:
                        continue
                    # too steep
                    if self.world[nny][nnx] - self.world[ny][nx] > 1:
                        continue
                    # occupied
                    if self.get_cell_player_idx([nnx, nny]) >= 0 and (nnx, nny) != (x, y):
                        continue
                    yield 'PUSH&BUILD', player_idx, (nx, ny), (nnx, nny)

            elif other_idx == -1:
                # free -> move
                for nnx, nny in self.iter_neighbors([nx, ny]):
                    # too steep
                    if self.world[ny][nx] - self.world[y][x] > 1: # TODO: move this code up
                        continue
                    # occupied
                    if self.get_cell_player_idx([nnx, nny]) >= 0 and (nnx, nny) != (x, y):
                        continue
                    yield 'MOVE&BUILD', player_idx, (nx, ny), (nnx, nny)

    def get_action(self):
        legal_actions = []
        # iterate over players, collecting all legal actions
        for player_idx in range(self.nb_players_per_side):
            legal_actions += self.iter_legal_actions(player_idx)
        if len(legal_actions) == 0:
            return 'ACCEPT-DEFEAT'
        else:
            return random.choice(legal_actions)

    def shutdown(self):
        pass
