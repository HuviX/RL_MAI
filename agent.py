import pickle
from collections import defaultdict

import numpy as np
from numpy import random


class BaseAgent:
    def __init__(self, sign: int, file_name: str):
        self.sign = sign
        self.file_name = file_name

    def get_action(self, env):
        raise NotImplementedError(
            "You can't call method from an abstract class itself"
        )

    def set_reward(self, reward: int, state, done: bool):
        raise NotImplementedError(
            "You can't call method from an abstract class itself"
        )


class RandomAgent(BaseAgent):
    def __init__(self, sign: int):
        self.sign = sign
        self.reward = 0.0
        self.amount_of_wins = 0

    def get_action(self, state):
        empty_cells, _ = state
        # print(empty_cells)
        action = self._explore(empty_cells)
        return (action, self.sign)

    def set_reward(self, reward):
        if reward == 1:
            self.amount_of_wins += 1
        self.reward += reward

    def _explore(self, empty_cells):
        move = np.random.choice(empty_cells)
        return move


class QAgent(BaseAgent):
    def __init__(
        self, sign: int, file_name: str, epsilon_policy, lr: float = 0.1
    ):
        self.sign = sign
        self.file_name = file_name
        self.epsilon_policy = epsilon_policy
        self.lr = lr
        self.decay_gamma = 0.9
        self.reward = 0.0
        self.amount_of_wins = 0
        self._init_q_matrix()

    def get_action(self, state) -> int:
        empty_cells, board_state = state
        threshold = self.epsilon_policy.get_epsilon()
        if random.uniform(0, 1) < threshold:
            action = self._explore(empty_cells, board_state)
        else:
            action = self._exploit(empty_cells, board_state)
        return (action, self.sign)

    def set_reward(self, reward: float):
        self.reward += reward
        if reward == 1:
            self.amount_of_wins += 1
        board_state = self.last_board_state
        last_move = self.last_move
        q_values = self.q_matrix[board_state]
        q_value = q_values[last_move]
        q_value += self.lr * (self.decay_gamma * reward - q_value)
        q_values[last_move] = q_value
        self.q_matrix[self.last_board_state] = q_values

    def _exploit(self, empty_cells: np.ndarray, board_state: str) -> int:
        q_values = self.q_matrix[board_state]
        possible_movements = q_values[empty_cells]
        movement = np.argmax(possible_movements)
        move = empty_cells[movement]
        self.last_move = move
        self.last_board_state = board_state
        return move

    def _explore(self, empty_cells: np.ndarray, board_state: str) -> int:
        move = np.random.choice(empty_cells)
        # print(move)
        self.last_move = move
        self.last_board_state = board_state
        return move

    def _init_q_matrix(self):
        self.q_matrix = defaultdict(lambda: np.zeros(9, dtype=float))

    def dump_q_matrix(self, filename: str):
        q_matrix = dict(self.q_matrix)
        with open(filename, "wb") as f:
            pickle.dump(q_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)


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
