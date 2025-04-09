import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import math
import random
import fenix


class Agent:
    def __init__(self, player: int, depth: int = 3):
        self.player = player
        self.depth = depth
        self.prev_actions = []
        self.thresh_midgame = 15
        self.thresh_lategame = 25

        self.mult = {
            "early": {
                "pieces": 1,
                "mobility": 0.1,
                "king_safety": 0.3,
                "king_threat": 0.5,
                "captures": 1,
            },
            "mid": {
                "pieces": 1,
                "mobility": 0.1,
                "king_safety": 2,
                "king_threat": 1,
                "captures": 2,
            },
            "late": {
                "pieces": 0.7,
                "mobility": 0.1,
                "king_safety": 0.5,
                "king_threat": 1,
                "captures": 1.2,
            },
        }

        self.score_contributions = {"pieces": [], "mobility": [], "king_safety": [], "king_threat": [], "captures": []}
        self.last_turn_scores = []

    def __str__(self):
        return "AlphaBetaEvaluate"
        return 'AlphaBeta2.0'

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

        more_depth = 0
        
        if len(state.actions()) <= 5:
            more_depth += 1

        print(f'    Analyse des possibilités avec une profondeur de {self.depth - 1 + more_depth}')
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
    """
    calcule une pronfondeur qui s'adapte en fonction 
    du nombre de pièces restantes sur le plateau et du temps restant

    diminuer la valeur absolue de k signifie augmenter la vitesse de l'exponentielle
    """
    def depth_calculator(self, pieces : int, board: tuple, ini_time ,remaining_time):
        print(str(pieces) + " " + str(board))
        ini_pieces = (board[0]-1)*(board[1]-1)

        depth = ini_pieces//pieces - 1

        # Moins il reste de temps, moins le facteur est grand
        time_factor = remaining_time/ini_time
        time_factor = max(0, min(1, time_factor))

        k = 1.6 #variable pour ajuster la vitesse de l'exponentielle
        depth_raw = math.floor(math.exp((ini_pieces / pieces) / k))-1

        depth = int(depth_raw * time_factor)

        #print("current depth = " + str(depth))
        return depth 
    

    # --- DEPTH CALCULATOR ---------------------------------------------------
    """
    calcule une pronfondeur qui s'adapte en fonction 
    du nombre de pièces restantes sur le plateau et du temps restant

    diminuer la valeur absolue de k signifie augmenter la vitesse de l'exponentielle
    """
    def depth_calculator(self, pieces : int, board: tuple, ini_time ,remaining_time):
        print(str(pieces) + " " + str(board))
        ini_pieces = (board[0]-1)*(board[1]-1)

        depth = ini_pieces//pieces - 1

        # Moins il reste de temps, moins le facteur est grand
        time_factor = remaining_time/ini_time
        time_factor = max(0, min(1, time_factor))

        k = 1.6 #variable pour ajuster la vitesse de l'exponentielle
        depth_raw = math.floor(math.exp((ini_pieces / pieces) / k))-1

        depth = int(depth_raw * time_factor)

        #print("current depth = " + str(depth))
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
        print(f"\n🧠 Ajustement des multiplicateurs — Victoire : {won}")
        
        for scores in self.last_turn_scores:
            for key, val in scores.items():
                self.score_contributions[key].append(val)

        if self.last_turn_scores:
            for key in self.mult["mid"]:
                avg_score = sum(self.score_contributions[key]) / len(self.score_contributions[key])
                adjustment = impact if avg_score > 0 else 1
                self.mult["mid"][key] *= adjustment
                print(f"    {key}: nouveau poids = {self.mult['mid'][key]:.3f}")
        
        self.last_turn_scores.clear()
        for key in self.score_contributions:
            self.score_contributions[key].clear()
        self.log_weights_to_file(won)



    def log_weights_to_file(self, won: bool, path="weights.txt"):
        """
        Enregistre les pondérations actuelles dans un fichier texte après une partie.
        Crée le fichier s'il n'existe pas.
        """
        weights = self.mult["mid"]
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"winner={1 if won else 0} ")
            f.write(" ".join([f"{k}={round(v, 3)}" for k, v in weights.items()]))
            f.write("\n")



    def _pieces_score(self, state, player):
        score = 0
        for pos, piece in state.pieces.items():
            value = piece
            mult = 1 if piece * player > 0 else -1
            base = value**2
            base += self._position_bonus(pos, player) * 0.5
            score += mult * base
        return score

    def _position_bonus(self, pos, player):
        row, col = pos
        dimension = (7, 8)
        is_center_bonus = (
            3 - abs((dimension[0] // 2) - row) - abs((dimension[1] // 2) - col)
        )
        territory_bonus = (
            0.5
            if (player == 1 and row > dimension[0] // 2)
            or (player == -1 and row < dimension[0] // 2)
            else 0
        )
        return is_center_bonus + territory_bonus

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
            fenix.FenixAction((5, 0), (4, 0), removed=frozenset()),
            fenix.FenixAction((1, 7), (2, 7), removed=frozenset()),
            fenix.FenixAction((1, 4), (1, 3), removed=frozenset()),
            fenix.FenixAction((5, 3), (5, 4), removed=frozenset()),
            fenix.FenixAction((3, 2), (3, 1), removed=frozenset()),
            fenix.FenixAction((2, 5), (2, 6), removed=frozenset()),
        ]
        return openings[turn]
