import numpy as np
from numpy import random
from collections import defaultdict


class BaseAgent:
    def __init__(self, sign: int, file_name: str):
        self.sign = sign
        self.file_name = file_name

    def get_action(self, env):
        raise NotImplementedError("You can't call method from an abstract class itself") # noqa

    def set_reward(self, reward: int, state, done: bool):
        raise NotImplementedError("You can't call method from an abstract class itself") # noqa


class RandomAgent(BaseAgent):
    def __init__(self, sign: int):
        self.sign = sign

    def get_action(self, empty_cells):
        print(empty_cells)
        action = self._explore(empty_cells)
        return action

    def set_reward(self, reward: float):
        pass

    def _explore(self, empty_cells):
        move = np.random.choice(empty_cells)
        return move

class QAgent(BaseAgent):
    def __init__(self, sign: int, file_name: str, epsilon_policy, lr: float = 0.001):
        self.sign = sign
        self.file_name = file_name
        self.epsilon_policy = epsilon_policy
        self.lr = lr
        self.decay_gamma = 0.9
        self._init_q_matrix()


    def get_action(self, empty_cells, board_state):
        threshold = self.epsilon_policy.get_epsilon()
        if random.uniform(0, 1) < threshold:
            action = self._explore(empty_cells)
        else:
            action = self._exploit(empty_cells, board_state)
        return action

    def set_reward(self, reward: float):
        board_state = self.last_board_state
        last_move = self.last_move
        q_values = self.q_matrix[board_state]
        q_value = q_values[last_move]
        q_value += self.lr * (self.decay_gamma * reward - q_value)
        q_values[last_move] = q_value
        self.q_matrix[self.last_board_state] = q_values
        

    def _exploit(self, empty_cells, board_state):
        q_values = self.q_matrix[board_state]
        # print(q_values)
        # print(empty_cells)
        possible_movements = q_values[empty_cells]
        movement = np.argmax(possible_movements)
        move = empty_cells[movement]
        self.last_move = move
        self.last_board_state = board_state
        return move

    def _explore(self, empty_cells, board_state):
        move = np.random.choice(empty_cells)
        # print(move)
        self.last_move = move
        self.last_board_state = board_state
        return move

    def _init_q_matrix(self):
        self.q_matrix = defaultdict(lambda: np.zeros(9, dtype=float))


class ConstantEpsilonFunction:
    def __init__(self, start_value: float):
        self.start_value = start_value

    def get_epsilon(self):
        return self.start_value

    @property
    def get_iter(self):
        return self.iter


class DecayEpsilonFunction:
    def __init__(self, start_value: float, min_value: float, mult: float):
        self.start_value = start_value
        self.min_value = min_value
        self.mult = mult
        self.iter = 0

    def get_epsilon(self):
        epsilon = np.exp(-self.iter * self.mult)
        self.iter += 1
        return max(self.min_value, epsilon)

    @property
    def get_iter(self):
        return self.iter
