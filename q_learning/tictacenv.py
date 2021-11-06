from typing import Tuple

import numpy as np


class TicTacEnvironment:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.is_over = False

    def reset(self) -> Tuple[np.ndarray, str]:
        """Resets an environment

        Returns:
            Tuple[np.ndarray, str]: empty cells and board state as a str
        """
        self.__init__()
        return self._get_state()

    def step(self, cell: int, sign: int) -> Tuple[int, np.ndarray, str]:
        """Gets action from agent. Fills cell with sign

        Args:
            cell: number of cell
            sign: sign to fill the chosen cell

        Returns:
            reward: float value of reward
            board_state as a tuple of (empty_cells, board_state)
            self.is_over: True if game is over otherwise False.
        """
        self._set_sign(cell, sign)
        reward = self._get_reward()
        state = self._get_state()
        return reward, state, self.is_over

    def _check_if_game_is_over(self) -> bool:
        """Checks if game is over.
        Gets info about all possible win combos.
        Gets amount of empty cells on a board

        Returns:
            bool: whether game is over or no
            int: amount of empty cells
        """
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

        if any(np.abs(combinations_sum) == 3):
            return True
        return False

    def _get_reward(self) -> int:
        """Returns the reward value and sets flag if game is over

        Returns:
            reward: reward value for an agent
        """
        win = self._check_if_game_is_over()
        empties = self._get_state().size

        if win:
            self.is_over = True
            reward = 1.0
        else:
            reward = 0.1
        if empties == 0:
            self.is_over = True
        return reward

    def _get_state(self) -> np.ndarray:
        """Return empty cells indices

        Returns:
            empty_cells: np.ndarray of empty cells indices
        """
        board = self.board.ravel()
        empty_cells = np.where(board == 0)[0]
        return empty_cells

    def _set_sign(self, cell: int, sign: int):
        """Sets sign to a given cell on a board"""
        board = self.board.ravel()
        board[cell] = sign
        self.board = board.reshape((3, 3))
