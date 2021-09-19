import numpy as np
from numpy import random
from collections import defaultdict


class BaseAgent:
    def __init__(self, sign: int, file_name: str):
        self.sign = sign
        self.file_name = file_name

    def get_action(self, env):
        raise NotImplementedError("You can't call method from an abstract class itself") # noqa

    def reward(self, reward: int, state, done: bool):
        raise NotImplementedError("You can't call method from an abstract class itself") # noqa


class QAgent(BaseAgent):
    def __init__(self, ):
        pass

    def make_action(self, empty_cells):
        
    def _init_q_matrix(self, ):
        self.q_matrix = defaultdict(lambda : np.zeros(9, dtype=float))