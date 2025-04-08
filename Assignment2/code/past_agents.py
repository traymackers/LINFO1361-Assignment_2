# -*- coding: utf-8 -*-
import math
import time
import random
from collections import defaultdict
import fenix
from agent import Agent


class RandomAgent(Agent):
    def act(self, state, remaining_time):
        actions = state.actions()
        if len(actions) == 0:
            raise Exception("No action available.")
        return random.choice(actions)

class Agent_first:
    def __init__(self, player: int, depth: int = 3):
        self.player = player
        self.depth = depth
        self.prev_actions = []

    def act(self, state, remaining_time):
        opening_moves = self._opening(state.turn)
        if opening_moves:
            return opening_moves

        # Plus de profondeur si le choix est isolé
        if state.turn > 10:
            my_king = [pos for pos, v in state.pieces.items() if v == 3 * self.player]
            if my_king:
                adjacent = sum(
                    1 for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                    if (dx != 0 or dy != 0)
                    and (my_king[0][0] + dx, my_king[0][1] + dy) in state.pieces
                    and state.pieces[(my_king[0][0] + dx, my_king[0][1] + dy)] * self.player > 0
                )
                if adjacent < 2:
                    self.depth += 1

        print(f'=== Tour n°{state.turn} ===')
        best_value = -math.inf
        best_action = None

        for action in self._ordered_actions(state):
            value = self._min_value(state, state.result(action), self.depth - 1, -math.inf, math.inf)
            if action in self.prev_actions:
                value -= 3

            if value > best_value or (value == best_value and random.randint(1, 8) == 1):
                best_value = value
                best_action = action
                print(f'Nouveau meilleur coup = {best_action}')

        self.prev_actions.append(best_action)
        if len(self.prev_actions) > 10:
            self.prev_actions.pop(0)

        print(f'Meilleur coup final = {best_action}')
        print(f'=== Fin du tour n°{state.turn} ===')
        return best_action

    # --- ÉVALUATION ---------------------------------------------------------

    def evaluate(self, actual_state: fenix.FenixState, next_state: fenix.FenixState, player: int) -> float:
        return (
            self._pieces_score(next_state, player) +
            self._mobility(next_state, player) * 0.1 +
            self._king_safety(next_state, player) * 0.5
        )

    def _pieces_score(self, state, player):
        score = 0
        dim = state.dim
        for pos, piece in state.pieces.items():
            value = abs(piece)
            mult = 1 if piece * player > 0 else -1
            base = value ** 2
            base += self._position_bonus(pos, dim, player) * 0.5
            if value == 3:  # Roi
                base += 2
            score += mult * base
        return score

    def _position_bonus(self, pos, dim, player):
        row, col = pos
        center_bonus = 3 - abs((dim[0] // 2) - row) - abs((dim[1] // 2) - col)
        territory_bonus = 1.0 if (player == 1 and row > dim[0] // 2) or (player == -1 and row < dim[0] // 2) else 0
        return center_bonus + territory_bonus

    def _mobility(self, state, player):
        temp = state.current_player
        state.current_player = player
        mobility = len(state.actions())
        state.current_player = temp
        return mobility

    def _king_safety(self, state, player):
        king_pos = [pos for pos, v in state.pieces.items() if v == 3 * player]
        if not king_pos:
            return -10
        row, col = king_pos[0]
        safe = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                npos = (row + dx, col + dy)
                if npos in state.pieces and state.pieces[npos] * player > 0:
                    safe += 1
        return safe

    # --- ALPHA-BETA --------------------------------------------------------

    def _max_value(self, actual_state, next_state, depth, alpha, beta):
        if next_state.is_terminal():
            return self._terminal_score(next_state)
        if depth == 0:
            return self.evaluate(actual_state, next_state, self.player)
        value = -math.inf
        for action in self._ordered_actions(next_state):
            value = max(value, self._min_value(actual_state, next_state.result(action), depth - 1, alpha, beta))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def _min_value(self, actual_state, next_state, depth, alpha, beta):
        if next_state.is_terminal():
            return self._terminal_score(next_state)
        if depth == 0:
            return self.evaluate(actual_state, next_state, self.player)
        value = math.inf
        for action in self._ordered_actions(next_state):
            value = min(value, self._max_value(actual_state, next_state.result(action), depth - 1, alpha, beta))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    def _terminal_score(self, state):
        util = state.utility(self.player)
        return 1000 * util

    # --- UTILITAIRES --------------------------------------------------------

    def _ordered_actions(self, state):
        # Trier les states à explorer en fonction de leur évaluation
        return sorted(state.actions(), key=lambda a: self.evaluate(state, state.result(a), self.player), reverse=True)

    def _opening(self, turn):
        # Place le roi dans le coin
        openings = [
            fenix.FenixAction((0, 1), (0, 0), removed=frozenset()),
            fenix.FenixAction((5, 7), (6, 7), removed=frozenset()),
            fenix.FenixAction((1, 0), (0, 0), removed=frozenset()),
            fenix.FenixAction((6, 6), (6, 7), removed=frozenset())
        ]
        return openings[turn]


all_agents = [RandomAgent, Agent_first,]