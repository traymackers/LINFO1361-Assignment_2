# -*- coding: utf-8 -*-
import math
import time
import random
from collections import defaultdict
import fenix

class Agent:
    def __init__(self, player: int, depth: int = 3, method: str = 'mcts'):
        self.player = player
        self.depth = depth
        self.method = method

    def act(self, state, remaining_time):
        if self.method == 'alpha-beta':
            return self._act_alphabeta(state)
        elif self.method == 'mcts':
            return self._act_mcts(state)
        else:
            raise ValueError(f"Méthode pas implémentée: {self.method}")

    # ---------- ALPHA-BETA ----------
    def _act_alphabeta(self, state):

        # Création du roi en (0,0) ou (6,7)
        if state.turn == 0:
            return fenix.FenixAction((0,1),(0,0), removed=frozenset())
        if state.turn == 1:
            return fenix.FenixAction((5,7),(6,7), removed=frozenset())
        if state.turn == 2:
            return fenix.FenixAction((1,0),(0,0), removed=frozenset())
        if state.turn == 3:
            return fenix.FenixAction((6,6),(6,7), removed=frozenset())
        

        #Calcul du meilleur coup
        best_value = -math.inf
        best_action = None
        print(f'=== Tour n°{state.turn} ===')
        for action in state.actions():
            print(f"Action possible: {action}")
            value = self._min_value(state.result(action), self.depth - 1, -math.inf, math.inf)
            if value > best_value:
                best_value = value
                best_action = action
                print(f'Nouveau meilleur coup = {best_action}')
        print(f'Meilleur coup final = {best_action}')
        print(f'Fin du tour n°{state.turn}')
        return best_action

    def _max_value(self, state, depth, alpha, beta):
        if state.is_terminal():
            return self._terminal_score(state)
        if depth == 0:
            return self.evaluate(state, self.player)
        value = -math.inf
        for action in state.actions():
            value = max(value, self._min_value(state.result(action), depth - 1, alpha, beta))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def _min_value(self, state, depth, alpha, beta):
        if state.is_terminal():
            return self._terminal_score(state)
        if depth == 0:
            return self.evaluate(state, self.player)
        value = math.inf
        for action in state.actions():
            value = min(value, self._max_value(state.result(action), depth - 1, alpha, beta))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    def _terminal_score(self, state):
        util = state.utility(self.player)
        return 1000 * util

    @staticmethod
    def evaluate(state: fenix.FenixState, player: int) -> float:
        score = 0
        for piece in state.pieces.values():
            if piece * player > 0:
                score += abs(piece)
            elif piece * player < 0:
                score -= abs(piece)
        return score

    # ---------- MCTS ----------
    def _act_mcts(self, root_state):

        if root_state.turn == 0:
            return fenix.FenixAction((0,1),(0,0), removed=frozenset())
        if root_state.turn == 1:
            return fenix.FenixAction((5,7),(6,7), removed=frozenset())
        if root_state.turn == 2:
            return fenix.FenixAction((1,0),(0,0), removed=frozenset())
        if root_state.turn == 3:
            return fenix.FenixAction((6,6),(6,7), removed=frozenset())

        stats = defaultdict(lambda: {"wins": 0, "visits": 0})
        end_time = time.time() + self.time_limit
        root_actions = root_state.actions()

        if not root_actions:
            raise Exception("No legal actions!")

        while time.time() < end_time:
            state = root_state
            path = []
            visited = set()

            # Sélection & Expansion
            while not state.is_terminal():
                actions = state.actions()
                if not actions:
                    break
                action = random.choice(actions)
                path.append((state, action))
                state = state.result(action)
                h = hash(state)
                if h not in stats:
                    break  # on développe un nouveau nœud

            # Simulation
            winner = self._rollout(state)

            # Backpropagation
            for (s, a) in path:
                h = hash(s)
                stats[h]["visits"] += 1
                if s.to_move() != winner:  # si c'était notre coup et on gagne
                    stats[h]["wins"] += 1

        # Choix final
        best_action = max(root_actions, key=lambda a: stats[hash(root_state.result(a))]["wins"])
        return best_action

    def _rollout(self, state):
        while not state.is_terminal():
            actions = state.actions()
            if not actions:
                break
            state = state.result(random.choice(actions))
        util = state.utility(self.player)
        if util > 0:
            return self.player
        elif util < 0:
            return -self.player
        return 0