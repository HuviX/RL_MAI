from typing import Tuple

import numpy as np

from agent import BaseAgent


class Environment:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.num_actions = 0
        self.is_over = False

    def get_board_hash(self):
        state = self.board.ravel().astype(str).tolist()
        state = "".join(state)
        return state

    def _check_if_game_is_over(self) -> Tuple[bool]:
        diagonal = self.board.diagonal()
        anti_diag = np.fliplr(self.board).diagonal()
        combinations_sum = np.abs(
            np.r_[
                diagonal.sum(),
                anti_diag.sum(),
                np.sum(self.board, axis=0),
                np.sum(self.board, axis=1),
            ]
        )
        board = self.board.ravel()
        # Проверим если какой-либо из игроков выиграл
        if any(np.abs(combinations_sum) == 3):
            return True, np.where(board == 0)[0].size
        # Если не выполняется, то игра продолжается
        return False, np.where(board == 0)[0].size

    def _feed_rewards(self, agent: BaseAgent):
        win, empties = self._check_if_game_is_over()
        if win:
            agent.set_reward(1.0)
            self.is_over = True
        else:
            agent.set_reward(0.1)
        if empties == 0:
            self.is_over = True

    def get_empty_cells(self):
        board = self.board.ravel()
        return np.where(board == 0)[0]

    def _set_sign(self, sign: int, cell: int):
        self.num_actions += 1
        board = self.board.ravel()
        board[cell] = sign
        self.board = board.reshape((3, 3))

    def set_action(self, agent: BaseAgent, cell: int):
        self._set_sign(agent.sign, cell)
        self._feed_rewards(agent)
