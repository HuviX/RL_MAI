from typing import Tuple, Union

import numpy as np


class TicTacEnvironment:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.num_actions = 0
        self.is_over = False

    def reset(self) -> Tuple[ np.ndarray, str]:
        self.__init__()
        return self._get_state()

    def step(self, cell: int, sign: int) -> Tuple[int, np.ndarray, str]:
        self._set_sign(cell, sign)
        reward = self._get_reward()
        empty_cells, board_state = self._get_state()
        return reward, (empty_cells, board_state), self.is_over

    def _get_state(self) -> Tuple[ np.ndarray, str]:
        empty_cells = self._get_empty_cells()
        board_state = self._get_board_state()
        return empty_cells, board_state

    def _get_board_state(self):
        state = self.board.ravel().astype(str).tolist()
        state = "".join(state)
        return state

    def _check_if_game_is_over(self) -> Tuple[bool, int]:
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

        if any(np.abs(combinations_sum) == 3):
            return True, np.where(board == 0)[0].size

        return False, np.where(board == 0)[0].size

    def _get_reward(self) -> int:
        win, empties = self._check_if_game_is_over()
        if win:
            self.is_over = True
            reward = 1.0
        else:
            reward = 0.1
        if empties == 0:
            self.is_over = True
        return reward

    def _get_empty_cells(self) -> np.ndarray:
        board = self.board.ravel()
        return np.where(board == 0)[0]

    def _set_sign(self, cell: int, sign: int):
        self.num_actions += 1
        board = self.board.ravel()
        board[cell] = sign
        self.board = board.reshape((3, 3))
