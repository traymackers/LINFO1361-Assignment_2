# -*- coding: utf-8 -*-
import math
import time
import random
from collections import defaultdict
import fenix
from agent import Agent


class RandomAgent:
    def __init__(self, player: int, depth: int = 3):
        self.player = player
        self.depth = depth

    def __str__(self):
        return "RandomAgent"
    def act(self, state, remaining_time):
        actions = state.actions()
        if len(actions) == 0:
            raise Exception("No action available.")
        return random.choice(actions)

class AlphaBeta:
    def __init__(self, player: int, depth: int = 3):
        self.player = player
        self.depth = depth

    def __str__(self):
        return "AlphaBeta"

    def act(self, state, remaining_time):
        best_value = -math.inf
        best_action = None
        for action in state.actions():
            value = self._min_value(state.result(action), self.depth - 1, -math.inf, math.inf)
            if value > best_value:
                best_value = value
                best_action = action
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

class AlphaBeta_MCTS:
    def __init__(self, player: int, depth: int = 3, method: str = 'mcts', time_limit: float = 1.5):
        self.player = player
        self.depth = depth
        self.method = method.lower()
        self.time_limit = time_limit

    def __str__(self):
        return "AlphaBeta_MCTS"

    def act(self, state, remaining_time):
        if self.method == 'alpha-beta':
            return self._act_alphabeta(state)
        elif self.method == 'mcts':
            return self._act_mcts(state)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    # ---------- ALPHA-BETA ----------
    def _act_alphabeta(self, state):

        if state.turn == 0:
            return fenix.FenixAction((0,1),(0,0), removed=frozenset())
        if state.turn == 1:
            return fenix.FenixAction((5,7),(6,7), removed=frozenset())
        if state.turn == 2:
            return fenix.FenixAction((1,0),(0,0), removed=frozenset())
        if state.turn == 3:
            return fenix.FenixAction((6,6),(6,7), removed=frozenset())

        best_value = -math.inf
        best_action = None
        for action in state.actions():
            value = self._min_value(state.result(action), self.depth - 1, -math.inf, math.inf)
            if value > best_value:
                best_value = value
                best_action = action
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

class AlphaBetaPlus:
    def __init__(self, player: int, depth: int = 3):
        self.player = player
        self.depth = depth
        self.prev_actions = []
    
    def __str__(self):
        return "AlphaBetaPlus"

    def act(self, state, remaining_time):
        if state.turn <= 3:
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

        best_value = -math.inf
        best_action = None

        for action in self._ordered_actions(state):
            value = self._min_value(state, state.result(action), self.depth - 1, -math.inf, math.inf)
            if action in self.prev_actions:
                value -= 3

            if value > best_value or (value == best_value and random.randint(1, 8) == 1):
                best_value = value
                best_action = action

        self.prev_actions.append(best_action)
        if len(self.prev_actions) > 10:
            self.prev_actions.pop(0)
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

class Alpha_variable_depth:
    def __init__(self, player: int, depth: int = 3):
        self.player = player
        self.depth = depth
        self.prev_actions = []
        self.thresh_midgame = 15
        self.thresh_lategame = 25

        self.mult = {
            "early": self._random_weights(),
            "mid": self._random_weights(),
            "late": self._random_weights(),
        }
        print(f'Partie d\'analyse avec ces poids: {self.mult}')

        self.score_contributions = {
            "pieces": [],
            "mobility": [],
            "king_safety": [],
            "king_threat": [],
            "captures": [],
        }
        self.last_turn_scores = []

    def _random_weights(self):
        return {
            "pieces": round(random.uniform(0, 3.0), 2),
            "mobility": round(random.uniform(0, 3), 3),
            "king_safety": round(random.uniform(0, 3), 2),
            "king_threat": round(random.uniform(0, 3), 2),
            "captures": round(random.uniform(0, 3), 2),
        }

    def __str__(self):
        return "Alpha_variable_depth2"

    def act(self, state, remaining_time):
        print(f"=== Tour n°{state.turn} ===")
        if state.turn == self.thresh_midgame or state.turn == self.thresh_midgame + 1:
            print(f"=== Passage en MidGame ===")
        if state.turn == self.thresh_lategame or state.turn == self.thresh_lategame + 1:
            print(f"=== Passage en LateGame ===")

        if state.turn <= 9:
            print(f"    Création des généraux et du roi (coin)")
            opening_moves = self._opening(state.turn)
            if opening_moves:
                print(f"        Coup final = {opening_moves}")
                return opening_moves

        best_value = -math.inf
        best_action = None

        total_pieces = state._has_piece(1) + state._has_piece(-1)
        more_depth = self.depth_calculator(total_pieces, state.dim, 300, remaining_time)

        # failsafe, diminue massivement le depth quand il ne reste plus que 20 secondes
        if remaining_time < 20 :
            more_depth = -1

        print(
            f"    Analyse des possibilités avec une profondeur de {self.depth - 1 + more_depth}"
        )
        for action in self._ordered_actions(state):
            value = self._opponent_turn_min(
                state,
                state.result(action),
                self.depth - 1 + more_depth,
                -math.inf,
                math.inf,
            )
            if action in self.prev_actions:
                value -= 3

            if value > best_value or (
                value == best_value and random.randint(1, 8) == 1
            ):
                best_value = value
                best_action = action
                print(
                    f"    Nouveau meilleur coup = {best_action} avec une valeur de {best_value}"
                )

        self.prev_actions.append(best_action)
        if len(self.prev_actions) > 10:
            self.prev_actions.pop(0)

        print(
            f"    Meilleur coup final = {best_action} avec une valeur de {best_value}"
        )
        return best_action

    # --- DEPTH CALCULATOR ---------------------------------------------------

    def depth_calculator(self, pieces: int, board: tuple, ini_time, remaining_time):
        print(str(pieces) + " " + str(board))
        ini_pieces = (board[0] - 1) * (board[1] - 1)

        depth = ini_pieces // pieces - 1

        # Moins il reste de temps, moins le facteur est grand
        time_factor = remaining_time / ini_time
        time_factor = max(0, min(1, time_factor))

        k = 3  # variable pour ajuster la vitesse de l'exponentielle
        depth_raw = math.floor(math.exp((ini_pieces / pieces) / k)) - 1

        depth = int(depth_raw * time_factor)

        # print("current depth = " + str(depth))
        return depth

    # --- ÉVALUATION ---------------------------------------------------------

    def evaluate(self, actual_state, next_state, player: int) -> float:
        if next_state.turn < self.thresh_midgame:
            mult = self.mult["early"]
        elif next_state.turn < self.thresh_lategame:
            mult = self.mult["mid"]
        else:
            mult = self.mult["late"]

        values = {
            "pieces": self._pieces_score(next_state, player),
            "mobility": self._mobility(next_state, player),
            "king_safety": self._king_safety(next_state, player),
            "king_threat": self._king_threat_score(next_state, player),
            "captures": self._multiple_capture(next_state, player),
        }

        total = 0
        for key in values:
            total += values[key] * mult[key]

        self.last_turn_scores.append(values)
        return round(total, 5)

    def update_multipliers_after_game(self, won: bool):
        impact = 1.05 if won else 0.95
        print(f"\n Ajout des poids — Victoire : {won}")

        for scores in self.last_turn_scores:
            for key, val in scores.items():
                self.score_contributions[key].append(val)

        if self.last_turn_scores:
            for key in self.mult["mid"]:
                avg_score = sum(self.score_contributions[key]) / len(
                    self.score_contributions[key]
                )
                adjustment = impact if avg_score > 0 else 1
                self.mult["mid"][key] *= adjustment
                print(f"    {key}: {self.mult['mid'][key]:.3f}")

        self.last_turn_scores.clear()
        for key in self.score_contributions:
            self.score_contributions[key].clear()
        self.log_weights_to_file(won)

    def log_weights_to_file(self, won: bool, path="weights.txt"):
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"winner={1 if won else 0} ")
            for phase in ["early", "mid", "late"]:
                for key, val in self.mult[phase].items():
                    f.write(f"{phase}_{key}={round(val, 3)} ")
            f.write("\n")

    def _pieces_score(self, state, player):
        score = 0
        for pos, piece in state.pieces.items():
            value = piece
            mult = 1 if piece * player > 0 else -1
            base = value**2
            score += mult * base
        return score
    
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

    def _king_threat_score(self, state, player):
        score = 0
        opponent_king = [pos for pos, v in state.pieces.items() if v == 3 * -player]
        my_pieces = [pos for pos, v in state.pieces.items() if v * player > 0]

        if opponent_king:
            for my_pos in my_pieces:
                dist = abs(my_pos[0] - opponent_king[0][0]) + abs(
                    my_pos[1] - opponent_king[0][1]
                )
                if dist <= 2:
                    score += 2  # tu peux ajuster
        return score

    def _multiple_capture(self, state, player):
        count_removed = 0
        for action in state.actions():
            if len(action.removed) > count_removed:
                count_removed = len(action.removed)

        return count_removed

    # --- ALPHA-BETA --------------------------------------------------------

    def _player_turn_max(self, actual_state, next_state, depth, alpha, beta):
        # Calcule le score d'un state (ou appelle le tour suivant)

        if next_state.is_terminal():
            return next_state.utility(self.player) * 1000

        if depth == 0:
            return self.evaluate(actual_state, next_state, self.player)

        score = -math.inf
        for action in self._ordered_actions(next_state):
            score = max(
                score,
                self._opponent_turn_min(
                    actual_state, next_state.result(action), depth - 1, alpha, beta
                ),
            )
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return score

    def _opponent_turn_min(self, actual_state, next_state, depth, alpha, beta):
        # Calcule le score d'un state (ou appelle le tour suivant)

        if next_state.is_terminal():
            return next_state.utility(self.player) * 1000

        if depth == 0:
            return self.evaluate(actual_state, next_state, self.player)

        score = math.inf
        for action in self._ordered_actions(next_state):
            score = min(
                score,
                self._player_turn_max(
                    actual_state, next_state.result(action), depth - 1, alpha, beta
                ),
            )
            if score <= alpha:
                return score
            beta = min(beta, score)
        return score

    # --- UTILITAIRES --------------------------------------------------------

    def _ordered_actions(self, state):
        # Trier les states à explorer en fonction de leur évaluation
        actions = state.actions()
        scores = []
        final_actions = []

        for action in actions:
            score = self.evaluate(state, state.result(action), self.player)
            scores.append((score, action))
        scores.sort(reverse=True)

        for action in scores:
            final_actions.append(action[1])

        return final_actions
    

    def _opening(self, turn):
        # Place le roi dans le coin (haut gauche ou bas droit)
        openings = [
            fenix.FenixAction((0, 1), (0, 0), removed=frozenset()),
            fenix.FenixAction((5, 7), (6, 7), removed=frozenset()),
            fenix.FenixAction((1, 0), (0, 0), removed=frozenset()),
            fenix.FenixAction((6, 6), (6, 7), removed=frozenset()),

            fenix.FenixAction((2, 0), (3, 0), removed=frozenset()),
            fenix.FenixAction((4, 7), (3, 7), removed=frozenset()),
            fenix.FenixAction((1, 1), (2, 1), removed=frozenset()),
            fenix.FenixAction((5, 6), (4, 6), removed=frozenset()),
            fenix.FenixAction((0, 2), (1, 2), removed=frozenset()),
            fenix.FenixAction((6, 5), (5, 5), removed=frozenset()),
        ] 
        return openings[turn]


class Alpha_no_depth:
    def __init__(self, player: int, depth: int = 3):
        self.player = player
        self.depth = depth
        self.prev_actions = []
        self.thresh_midgame = 15
        self.thresh_lategame = 25

        self.mult = {
            "early": self._random_weights(),
            "mid": self._random_weights(),
            "late": self._random_weights(),
        }
        print(f'Partie d\'analyse avec ces poids: {self.mult}')

        self.score_contributions = {
            "pieces": [],
            "mobility": [],
            "king_safety": [],
            "king_threat": [],
            "captures": [],
        }
        self.last_turn_scores = []

    def _random_weights(self):
        return {
            "pieces": round(random.uniform(0, 3.0), 2),
            "mobility": round(random.uniform(0, 3), 3),
            "king_safety": round(random.uniform(0, 3), 2),
            "king_threat": round(random.uniform(0, 3), 2),
            "captures": round(random.uniform(0, 3), 2),
        }

    def __str__(self):
        return "Alpha_no_depth2"

    def act(self, state, remaining_time):
        print(f"=== Tour n°{state.turn} ===")
        if state.turn == self.thresh_midgame or state.turn == self.thresh_midgame + 1:
            print(f"=== Passage en MidGame ===")
        if state.turn == self.thresh_lategame or state.turn == self.thresh_lategame + 1:
            print(f"=== Passage en LateGame ===")

        if state.turn <= 9:
            print(f"    Création des généraux et du roi (coin)")
            opening_moves = self._opening(state.turn)
            if opening_moves:
                print(f"        Coup final = {opening_moves}")
                return opening_moves

        best_value = -math.inf
        best_action = None

        #total_pieces = state._has_piece(1) + state._has_piece(-1)
        #more_depth = self.depth_calculator(total_pieces, state.dim, 300, remaining_time)
        more_depth = 0

        # failsafe, diminue massivement le depth quand il ne reste plus que 20 secondes
        if remaining_time < 20 :
            more_depth = -1

        print(
            f"    Analyse des possibilités avec une profondeur de {self.depth - 1 + more_depth}"
        )
        for action in self._ordered_actions(state):
            value = self._opponent_turn_min(
                state,
                state.result(action),
                self.depth - 1 + more_depth,
                -math.inf,
                math.inf,
            )
            if action in self.prev_actions:
                value -= 3

            if value > best_value or (
                value == best_value and random.randint(1, 8) == 1
            ):
                best_value = value
                best_action = action
                print(
                    f"    Nouveau meilleur coup = {best_action} avec une valeur de {best_value}"
                )

        self.prev_actions.append(best_action)
        if len(self.prev_actions) > 10:
            self.prev_actions.pop(0)

        print(
            f"    Meilleur coup final = {best_action} avec une valeur de {best_value}"
        )
        return best_action

    # --- DEPTH CALCULATOR ---------------------------------------------------

    def depth_calculator(self, pieces: int, board: tuple, ini_time, remaining_time):
        print(str(pieces) + " " + str(board))
        ini_pieces = (board[0] - 1) * (board[1] - 1)

        depth = ini_pieces // pieces - 1

        # Moins il reste de temps, moins le facteur est grand
        time_factor = remaining_time / ini_time
        time_factor = max(0, min(1, time_factor))

        k = 2  # variable pour ajuster la vitesse de l'exponentielle
        depth_raw = math.floor(math.exp((ini_pieces / pieces) / k)) - 1

        depth = int(depth_raw * time_factor)

        # print("current depth = " + str(depth))
        return depth

    # --- ÉVALUATION ---------------------------------------------------------

    def evaluate(self, actual_state, next_state, player: int) -> float:
        if next_state.turn < self.thresh_midgame:
            mult = self.mult["early"]
        elif next_state.turn < self.thresh_lategame:
            mult = self.mult["mid"]
        else:
            mult = self.mult["late"]

        values = {
            "pieces": self._pieces_score(next_state, player),
            "mobility": self._mobility(next_state, player),
            "king_safety": self._king_safety(next_state, player),
            "king_threat": self._king_threat_score(next_state, player),
            "captures": self._multiple_capture(next_state, player),
        }

        total = 0
        for key in values:
            total += values[key] * mult[key]

        self.last_turn_scores.append(values)
        return round(total, 5)

    def update_multipliers_after_game(self, won: bool):
        impact = 1.05 if won else 0.95
        print(f"\n Ajout des poids — Victoire : {won}")

        for scores in self.last_turn_scores:
            for key, val in scores.items():
                self.score_contributions[key].append(val)

        if self.last_turn_scores:
            for key in self.mult["mid"]:
                avg_score = sum(self.score_contributions[key]) / len(
                    self.score_contributions[key]
                )
                adjustment = impact if avg_score > 0 else 1
                self.mult["mid"][key] *= adjustment
                print(f"    {key}: {self.mult['mid'][key]:.3f}")

        self.last_turn_scores.clear()
        for key in self.score_contributions:
            self.score_contributions[key].clear()
        self.log_weights_to_file(won)

    def log_weights_to_file(self, won: bool, path="weights.txt"):
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"winner={1 if won else 0} ")
            for phase in ["early", "mid", "late"]:
                for key, val in self.mult[phase].items():
                    f.write(f"{phase}_{key}={round(val, 3)} ")
            f.write("\n")

    def _pieces_score(self, state, player):
        score = 0
        for pos, piece in state.pieces.items():
            value = piece
            mult = 1 if piece * player > 0 else -1
            base = value**2
            score += mult * base
        return score
    
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

    def _king_threat_score(self, state, player):
        score = 0
        opponent_king = [pos for pos, v in state.pieces.items() if v == 3 * -player]
        my_pieces = [pos for pos, v in state.pieces.items() if v * player > 0]

        if opponent_king:
            for my_pos in my_pieces:
                dist = abs(my_pos[0] - opponent_king[0][0]) + abs(
                    my_pos[1] - opponent_king[0][1]
                )
                if dist <= 2:
                    score += 2  # tu peux ajuster
        return score

    def _multiple_capture(self, state, player):
        count_removed = 0
        for action in state.actions():
            if len(action.removed) > count_removed:
                count_removed = len(action.removed)

        return count_removed

    # --- ALPHA-BETA --------------------------------------------------------

    def _player_turn_max(self, actual_state, next_state, depth, alpha, beta):
        # Calcule le score d'un state (ou appelle le tour suivant)

        if next_state.is_terminal():
            return next_state.utility(self.player) * 1000

        if depth == 0:
            return self.evaluate(actual_state, next_state, self.player)

        score = -math.inf
        for action in self._ordered_actions(next_state):
            score = max(
                score,
                self._opponent_turn_min(
                    actual_state, next_state.result(action), depth - 1, alpha, beta
                ),
            )
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return score

    def _opponent_turn_min(self, actual_state, next_state, depth, alpha, beta):
        # Calcule le score d'un state (ou appelle le tour suivant)

        if next_state.is_terminal():
            return next_state.utility(self.player) * 1000

        if depth == 0:
            return self.evaluate(actual_state, next_state, self.player)

        score = math.inf
        for action in self._ordered_actions(next_state):
            score = min(
                score,
                self._player_turn_max(
                    actual_state, next_state.result(action), depth - 1, alpha, beta
                ),
            )
            if score <= alpha:
                return score
            beta = min(beta, score)
        return score

    # --- UTILITAIRES --------------------------------------------------------

    def _ordered_actions(self, state):
        # Trier les states à explorer en fonction de leur évaluation
        actions = state.actions()
        scores = []
        final_actions = []

        for action in actions:
            score = self.evaluate(state, state.result(action), self.player)
            scores.append((score, action))
        scores.sort(reverse=True)

        for action in scores:
            final_actions.append(action[1])

        return final_actions
    

    def _opening(self, turn):
        # Place le roi dans le coin (haut gauche ou bas droit)
        openings = [
            fenix.FenixAction((0, 1), (0, 0), removed=frozenset()),
            fenix.FenixAction((5, 7), (6, 7), removed=frozenset()),
            fenix.FenixAction((1, 0), (0, 0), removed=frozenset()),
            fenix.FenixAction((6, 6), (6, 7), removed=frozenset()),

            fenix.FenixAction((2, 0), (3, 0), removed=frozenset()),
            fenix.FenixAction((4, 7), (3, 7), removed=frozenset()),
            fenix.FenixAction((1, 1), (2, 1), removed=frozenset()),
            fenix.FenixAction((5, 6), (4, 6), removed=frozenset()),
            fenix.FenixAction((0, 2), (1, 2), removed=frozenset()),
            fenix.FenixAction((6, 5), (5, 5), removed=frozenset()),
        ] 
        return openings[turn]
    
all_agents = [RandomAgent, AlphaBeta, AlphaBeta_MCTS, AlphaBetaPlus,Alpha_variable_depth, Alpha_no_depth]