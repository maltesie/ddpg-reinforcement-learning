''' Module containing the game mechanics of https://www.codingame.com/multiplayer/bot-programming/wondev-woman
   * a test game is run step-wise when this file is executed
   * different AIs can be imported below
'''

#!/usr/bin/env python3

import random
import rl_agent as ai1
import random_agent as ai2
from collections import Counter

class Player(object):
    '''stores position and team of a figure'''
    def __init__(self, team, position=None):
        assert(len(str(team)) == 1) # teamname is one char
        self.team = team
        self.position = None
        if position is not None:
            self.set_position(position)

    def set_position(self, position):
        self.position = list(position)


class World(object):
    '''contains the world grid and the figures'''
    def __init__(self, size, nb_players_per_side):
        '''inits a quadratic world grid with sidelength `size` and `nb_players_per_side` players per team'''
        self.map = [[0 for _ in range(size)] for _ in range(size)]
        self.players = [Player('x') for _ in range(nb_players_per_side)] + [Player('o') for _ in range(nb_players_per_side)]
        for p in self.players:
            # figure out random un-occupied position
            while True:
                pos = random.randrange(size), random.randrange(size)
                if self.get_cell_player_idx(pos) == -1:
                    break
            p.set_position(pos)

    def get_cell_player_idx(self, position):
        '''returns the player index if there is a player blocking the cell, -1 otherwise'''
        position = list(position)
        for i, player in enumerate(self.players):
            if player.position == position:
                return i
        return -1

    def get_step_position(self, source, direction):
        '''converts a geographic / compass direction to an actual position'''
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        if direction not in directions:
            # no need to convert
            return direction

        offsets = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
        dx, dy = offsets[directions.index(direction)]
        x, y = source
        return x + dx, y + dy

    def is_legal_action(self, action_type, player_idx, dir1, dir2):
        '''returns whether an action is legal'''
        # translate compass directions to positions if needed
        pos1 = self.get_step_position(self.players[player_idx].position, dir1)
        pos2 = self.get_step_position(pos1, dir2)

        x, y = self.players[player_idx].position
        nx, ny = pos1
        nnx, nny = pos2

        # o.o.b
        if (nx < 0 or ny < 0 or nx >= len(self.map) or ny >= len(self.map) or
                nnx < 0 or nny < 0 or nnx >= len(self.map) or nny >= len(self.map)):
            return False

        # too far
        if abs(x-nx) > 1 or abs(y-ny) > 1 or abs(nx-nnx) > 1 or abs(ny-nny) > 1:
            return False

        if action_type == 'PUSH&BUILD':
            # illegal push direction
            if abs((nx - x) - (nnx - nx)) + abs((ny - y) - (nny - ny)) > 1:
                return False
            # too steep
            if self.map[nny][nnx] - self.map[ny][nx] > 1:
                return False
            # no enemy to push
            other_idx = self.get_cell_player_idx([nx, ny])
            if other_idx == -1 or self.players[player_idx].team == self.players[other_idx].team:
                return False
            # target occupied
            if self.get_cell_player_idx([nnx, nny]) >= 0: # and (nnx, nny) != (x, y)
                return False

        elif action_type == 'MOVE&BUILD':
            # too steep
            if self.map[ny][nx] - self.map[y][x] > 1:
                return False
            # illegal build
            if self.map[nny][nnx] < 0 or self.map[nny][nnx] > 3:
                return False
            # occupied
            if self.get_cell_player_idx([nnx, nny]) >= 0 and (nnx, nny) != (x, y):
                return False

        # unknwon action type
        else:
            return False

        # is legal action
        return True

    def apply_action(self, action_type, player_idx, dir1, dir2):
        '''manipulates the world with an action. returns score gain/loss per team.'''
        # translate compass directions to positions if needed
        pos1 = self.get_step_position(self.players[player_idx].position, dir1)
        pos2 = self.get_step_position(pos1, dir2)

        if action_type == 'PUSH&BUILD':
            pushed_player_idx = self.get_cell_player_idx(pos1)
            if self.get_cell_player_idx(pos2) == -1: # TODO: is there another case? maybe due to board visibility...
                self.players[pushed_player_idx].set_position(pos2)
                x, y = pos1
                self.map[y][x] = self.map[y][x] + 1

        elif action_type == 'MOVE&BUILD':
            self.players[player_idx].set_position(pos1)
            if self.get_cell_player_idx(pos2) == -1: # TODO: is there another case? maybe due to board visibility...
                x, y = pos2
                self.map[y][x] = self.map[y][x] + 1
                if self.map[y][x] == 3:
                    return Counter({self.players[player_idx].team: 1}) # 1up!

        else:
            raise RuntimeError('unknown action type')

        return Counter()

    def get_grid_rows(self):
        '''returns the world grid as in the online game'''
        return [''.join(['.' if self.map[y][x] == 4 else str(self.map[y][x]) for x in range(len(self.map))]) for y in range(len(self.map))]

    def __repr__(self):
        #map_matrix = [[0 if self.map[y][x] == 4 else self.map[y][x] + 1 for x in range(len(self.map))] for y in range(len(self.map))]
        player_matrix = [[' ' for _ in range(len(self.map))] for _ in range(len(self.map))]
        for p in self.players:
            x, y = p.position
            player_matrix[y][x] = str(p.team)

        return '\n'.join([' '.join([player_matrix[y][x] + str(self.map[y][x]) for x in range(len(self.map))]) for y in range(len(self.map))])


class Game(object):
    '''contains the world, the playing agents and the game mechanics'''
    def __init__(self, size, nb_players_per_side, agent1, agent2):
        self.world = World(size, nb_players_per_side)
        self.teams = ['x', 'o']
        self.active_team = self.teams[0]
        self.agents = {}
        self.agents[self.teams[0]] = agent1
        self.agents[self.teams[1]] = agent2
        self.team_disabled = { self.teams[0]: False, self.teams[1]: False }
        self.scores = Counter({ t: 0 for t in self.teams })
        self.over = False

    def get_winning_team(self):
        '''returns x, o or None'''
        if self.scores[self.teams[0]] > self.scores[self.teams[1]]:
            return self.teams[0]
        elif self.scores[self.teams[1]] > self.scores[self.teams[0]]:
            return self.teams[1]
        else:
            return None

    def turn(self):
        '''performs a game step'''
        team = self.active_team
        other_team = self.teams[1] if self.teams[0] == self.active_team else self.teams[0]
        agent = self.agents[team]
        # supply the agent with the state (and rewards of the previous action)
        agent.set_world(self.world.get_grid_rows(), (self.scores[team], self.scores[other_team]))
        # sort players by active team
        players = [p for p in self.world.players if p.team == team] + [p for p in self.world.players if p.team != team]
        for i, player in enumerate(players):
            agent.players[i].set_position(player.position)

        action = agent.get_action()

        if action == 'ACCEPT-DEFEAT':
            self.team_disabled[team] = True
        else:
            # translate client-side player id back to server-side player id
            action = list(action)
            action[1] = self.world.players.index(players[action[1]])

            if self.world.is_legal_action(*action):
                score_gain = self.world.apply_action(*action)
                self.scores += score_gain
            else:
                # requesting an illegal action ends the game
                self.over = True

        # check if game is over
        if (all(self.team_disabled.values()) or
                (self.team_disabled[self.teams[0]] and self.get_winning_team() == self.teams[1]) or
                (self.team_disabled[self.teams[1]] and self.get_winning_team() == self.teams[0]) or
                self.over):
            self.over = True
        else:
            # switch active team
            for t in self.teams:
                if t != self.active_team and not self.team_disabled[t]:
                    self.active_team = t
                    break

        return team, action

    def print_state(self):
        print(self.scores)
        print(self.world)

if __name__ == '__main__':
    agent1 = ai1.Agent(2)
    agent2 = ai1.Agent(2)
    game = Game(5, 2, agent1, agent2)
    while not game.over:
        game.print_state()
        input()
        print(game.turn())
    agent1.shutdown()
    agent2.shutdown()
