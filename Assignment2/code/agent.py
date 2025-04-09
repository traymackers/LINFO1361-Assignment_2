import sys

sys.stdout.reconfigure(encoding="utf-8")
import math
import random
import fenix


class Agent:
    def __init__(self, player: int, depth: int = 3):
        self.player = player
        self.depth = depth #profondeur de recherche
        self.prev_actions = [] #actions précédentes

        # TODO - Bien pondérer en fonction du temps et de la partie
        self.mult = {
            "pieces": 1.756,
            "mobility": 1.341,
            "king_safety": 0.3,
            "king_threat": 1.699,
            "captures": 1.213} #pondération d'évaluation


    def __str__(self):
        return "Finalpha"

    def act(self, state, remaining_time):
        print(f"=== Tour n°{state.turn} ===")

        if state.turn <= 9: #disposition prédéfinie
            print(f"    Création des généraux et du roi (coin)")
            predifined_first_moves = self._opening(state.turn)
            if predifined_first_moves:
                print(f"        Coup final = {predifined_first_moves}")
                return predifined_first_moves

        best_value = -math.inf
        best_action = None

        more_depth = 0 #bonus de profondeur
        total_pieces_on_board = state._has_piece(1) + state._has_piece(-1)
        more_depth = self.depth_calculator(total_pieces_on_board, state.dim, 300, remaining_time)

        # failsafe, diminue massivement le depth quand il ne reste plus que 20 secondes
        if remaining_time < 20 :
            self.depth = 2
            more_depth = 0

        print(
            f"    Analyse des possibilités avec une profondeur de {self.depth - 1 + more_depth}"
        )
        print(f'    Nombre de coups possibles: {len(state.actions())}')
        for action in self._ordered_actions(state):
            value = self._opponent_turn_min(
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
        """
        calcule une pronfondeur qui s'adapte en fonction
        du nombre de pièces restantes sur le plateau et du temps restant

        diminuer la valeur absolue de k signifie augmenter la vitesse de l'exponentielle
        """
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

    
    def evaluate(self, next_state, player: int) -> float:
        values = {
            "pieces": self._pieces_score(next_state, player),
            "mobility": self._mobility(next_state, player),
            "king_safety": self._king_safety(next_state, player),
            "king_threat": self._king_threat_score(next_state, player),
            "captures": self._multiple_capture(next_state),
        }

        total = 0
        for key in values:
            total += values[key] * self.mult[key]

        return round(total, 5)
    

    def _pieces_score(self, state, player):
        score = 0
        for pos, piece in state.pieces.items():
            owner_piece_value = 1 if piece * player > 0 else -1
            base = abs(piece)
            if base == 3:
                value = 100  # roi
            elif base == 2:
                value = 10   # général
            else:
                value = 1    # soldat
            score += owner_piece_value * value #si le pion est allié ou ennemi
        return score


    def _mobility(self, state, player):
        '''Calcule le nombre d'actions possibles
        '''
        temp = state.current_player
        state.current_player = player
        moves = state.actions()
        state.current_player = temp
        return len(moves)

    def _king_safety(self, state, player):
        '''Calcule si le roi est en sécurité
        '''
        king_pos = []
        for pos, v in state.pieces.items():
            if v == 3 * player:
                king_pos.append(pos)
        if king_pos == []:
            return -50 #malus -> roi disparu
        row, col = king_pos[0]
        score = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                npos = (row + dx, col + dy)
                if npos in state.pieces and state.pieces[npos] * player > 0:
                    score += 1
        # Bonus si le roi est dans un coin
        if (row in [0, state.dim[0]-1]) and (col in [0, state.dim[1]-1]):
            score += 2
        return score

    def _king_threat_score(self, state, player):
        '''Calcule si le roi adverse est menacé (proche d'un pion)
        '''
        score = 0

        opponent_king = []
        for pos, v in state.pieces.items():
            if v == 3 * -player:
                opponent_king.append(pos)
        
        my_pieces = []
        for pos, v in state.pieces.items():
            if v * player > 0:
                my_pieces.append((pos, abs(v)))

        if opponent_king != []:
            kx, ky = opponent_king[0]
            for (px, py), ptype in my_pieces:
                dist = abs(px - kx) + abs(py - ky)
                if dist <= 2:
                    bonus = 2
                    if ptype == 2:
                        bonus = 3
                    elif ptype == 3:
                        bonus = 5
                    score += bonus
        return score

    def _multiple_capture(self, state):
        '''Calcule s'il est possible de faire plusieurs captures (multi-kill)
        '''
        max_removed_value = 0
        for action in state.actions():
            total_value = sum(abs(state.pieces.get(pos, 0)) for pos in action.removed)
            if total_value > max_removed_value:
                max_removed_value = total_value
        return max_removed_value

    # --- ALPHA-BETA --------------------------------------------------------

    def _player_turn_max(self, next_state, depth, alpha, beta):
        # Calcule le score d'un state (ou appelle le tour suivant) - max donc plus score est haut, mieux c'est

        if next_state.is_terminal():
            return next_state.utility(self.player) * 1000

        if depth == 0:
            return self.evaluate(next_state, self.player)

        score = -math.inf
        for action in self._ordered_actions(next_state):
            score = max(
                score,
                self._opponent_turn_min(next_state.result(action), depth - 1, alpha, beta
                ),
            )
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return score

    def _opponent_turn_min(self, next_state, depth, alpha, beta):
        # Calcule le score d'un state (ou appelle le tour suivant) - min donc plus score est bas, mieux c'est

        if next_state.is_terminal():
            return next_state.utility(self.player) * 1000

        if depth == 0:
            return self.evaluate(next_state, self.player)

        score = math.inf
        for action in self._ordered_actions(next_state):
            score = min(
                score,
                self._player_turn_max(
                    next_state.result(action), depth - 1, alpha, beta
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
            score = self.evaluate(state.result(action), self.player)
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