import sys
sys.stdout.reconfigure(encoding='utf-8')
import math
import random
import fenix

class Agent:
    def __init__(self, player: int, depth: int = 3):
        self.player = player
        self.depth = depth
        self.prev_actions = []

    def __str__(self):
        return 'AlphaBetaEvaluate'

    def act(self, state, remaining_time):
        print(f'=== Tour n°{state.turn} ===')
        if state.turn <= 3:
            print(f'    Création du roi dans un coin')
            opening_moves = self._opening(state.turn)
            if opening_moves:
                print(f'        Coup final = {opening_moves}')
                return opening_moves

        best_value = -math.inf
        best_action = None

        more_depth = 0
        
        if len(state.actions()) <= 5:
            more_depth += 1

        print(f'    Analyse des possibilités avec une profondeur de {self.depth - 1 + more_depth}')
        for action in self._ordered_actions(state):
            value = self._opponent_turn_min(state, state.result(action), self.depth - 1 + more_depth, -math.inf, math.inf)
            if action in self.prev_actions:
                value -= 3

            if value > best_value or (value == best_value and random.randint(1, 8) == 1):
                best_value = value
                best_action = action
                print(f'    Nouveau meilleur coup = {best_action} avec une valeur de {best_value}')

        self.prev_actions.append(best_action)
        if len(self.prev_actions) > 10:
            self.prev_actions.pop(0)

        print(f'    Meilleur coup final = {best_action} avec une valeur de {best_value}')
        return best_action

    # --- ÉVALUATION ---------------------------------------------------------

    def evaluate(self, actual_state: fenix.FenixState, next_state: fenix.FenixState, player: int) -> float:
        
        score_pieces = self._pieces_score(next_state, player) * 1
        score_mobility = self._mobility(next_state, player) * 0.1
        score_king_safety = self._king_safety(next_state, player) * 0.5

        total_score = score_pieces + score_mobility + score_king_safety

        if random.randint(1,25000) == 666:
            print(f'        Évaluation du tour n°{next_state.turn}')
            print(f'            Score pièce:        {score_pieces}')
            print(f'            Score mobilité:     {score_mobility}')
            print(f'            Score sécurité roi: {score_king_safety}')
            print(f'            Score total:        {total_score}')
        
        return total_score

    def _pieces_score(self, state, player):
        score = 0
        dim = state.dim
        for pos, piece in state.pieces.items():
            value = abs(piece)
            mult = 1 if piece * player > 0 else -1
            base = value ** 2
            base += self._position_bonus(pos, dim, player) * 0.5
            if value == 3:
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

    def _player_turn_max(self, actual_state, next_state, depth, alpha, beta):
        # Calcule le score d'un state (ou appelle le tour suivant)

        if next_state.is_terminal():
            return next_state.utility(self.player) * 1000
        
        if depth == 0:
            return self.evaluate(actual_state, next_state, self.player)
        
        score = -math.inf
        for action in self._ordered_actions(next_state):
            score = max(score, self._opponent_turn_min(actual_state, next_state.result(action), depth - 1, alpha, beta))
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
            score = min(score, self._player_turn_max(actual_state, next_state.result(action), depth - 1, alpha, beta))
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
            fenix.FenixAction((6, 6), (6, 7), removed=frozenset())
        ]
        return openings[turn]
