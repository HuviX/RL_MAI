from agent import BaseAgent
from typing import List, Tuple

from collections import defaultdict
import numpy as np
from numpy import random

from agent import BaseAgent


class BaseEnvironment:
    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents
        self.board = np.zeros((3, 3), dtype=int)

    @property
    def print_board(self):
        print(self.board)

    def get_board_hash(self):
        state = self.board.ravel().astype(str).tolist()
        state = "".join(state)
        return hash(state)
    
    
    def _check_if_game_is_over(self) -> Tuple[bool]:
        diagonal = self.board.diagonal()
        anti_diag = np.fliplr(self.board).diagonal()
        combinations_sum = (
            np.abs(
                np.r_[
                    diagonal.sum(),
                    anti_diag.sum(),
                    np.sum(self.board, axis=0),
                    np.sum(self.board, axis=1)
                ]
            )
        )
        # Проверим если какой-либо из игроков выиграл
        if any(np.abs(combinations_sum) == 3):
            return True, True
        # Проверим если доска закончилась
        if np.where(self.board == 0)[0].size == 0:
            return True, False
        # Если ничего выше не выполняется, то игра продолжается
        return False, False
