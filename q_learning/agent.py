import pickle
from collections import defaultdict
from typing import Tuple

import numpy as np
from numpy import random


class BaseAgent:
    def __init__(self, what_figure, state, file_name):
        # чем ходим what_figure = 1 - крестик , 0 - нолик
        # state - список списков - доска 3x3 [[-1,-1,-1], [-1,-1,-1], [-1,-1,-1]]  # noqa
        #                           крестик - 1, нолик - 0, пусто - -1
        # агент сохраняет необходимую информацию в файл c именем file_name
        raise NotImplementedError(
            "You can't call method from an abstract class itself"
        )

    def get_action(self):
        raise NotImplementedError(
            "You can't call method from an abstract class itself"
        )

    def set_reward_and_state(self, reward, state, done):
        # reward - 0 - не выиграл, 1- выиграл
        # done - 0 - игра продолжается, 1 - игра окончена
        raise NotImplementedError(
            "You can't call method from an abstract class itself"
        )


class RandomAgent(BaseAgent):
    def __init__(self, sign: int, state: np.ndarray):
        """Creates agent instance.
        This agent movements are random all the time

        Args:
            sign: sign to play (-1 or 1)
        """
        self.sign = sign
        self.reward = 0.0
        self.amount_of_wins = 0
        self.state = state

    def set_reward_and_state(self, reward, state, done):
        self.set_reward(reward)
        self.done = done
        self.state = state

    def get_action(self, *args):
        return self._get_action(self.state)

    def _get_action(self, empty_cells, *args):
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
        self,
        sign: int,
        filename: str,
        state: np.ndarray,
        epsilon_policy=None,
        lr: float = 0.1,
        discount: float = 0.8,
    ):
        """Creates agent instance. Agent movements random are not random
        all the time. Randomness controls by an `epsilon_policy` arg.
        link: `https://en.wikipedia.org/wiki/Q-learning`

        Args:
            sign: sign to play (-1 or 1)
            file_name: file_name to store q matrix after learning
            epsilon_policy: Epsilon Policy class instance. Controls
                exploration/exploitation policy.
            lr: learning rate for q learning
            discount: discount (gamma) factor in q learning formula
        """
        self.sign = sign
        self.filename = filename
        self.lr = lr
        self.discount = discount
        self.reward = 0.0
        self.amount_of_wins = 0
        self.state = state

        if not epsilon_policy:
            epsilon_policy = ConstantEpsilonFunction()
        self.epsilon_policy = epsilon_policy
        self._init_q_matrix()

    def set_reward_and_state(self, reward, state, done):
        self.set_reward(reward)
        self.done = done
        self.state = state

    def get_action(self, epsilon_value: float = None):
        return self._get_action(self.state, epsilon_value)

    def _get_action(
        self,
        state: np.ndarray,
        policy_value: float = None,
    ) -> Tuple[int, int]:
        """Returns an action based on a board state and empty cells
            of environment. Decides wheter agent need to play randomly or
            exploit its knowledge.
        Args:
            state: list of empty cells on a tic tac toe board
            policy_value: float value for testing purposes. Set this to 1
                if you want to always get exploitation.
        Returns:
            action: index of cell
            sign: sign to be placed on an action cell
        """
        board_state = state.ravel().astype(str).tolist()
        board_state = "".join(board_state)

        threshold = policy_value or self.epsilon_policy.get_epsilon()
        if random.uniform(0, 1) < threshold:
            action = self._exploit(state, board_state)
        else:
            action = self._explore(state, board_state)
        return (action, self.sign)

    def set_reward(self, reward: float):
        """Performs reward processing and learning process

        Args:
            reward: reward value
        """
        self.reward += reward
        if reward == 1:
            self.amount_of_wins += 1
        board_state = self.last_board_state
        last_move = self.last_move
        q_values = self.q_matrix[board_state]
        q_value = q_values[last_move]
        max_q_value = np.max(q_values)
        new_q_value = (1 - self.lr) * q_value + self.lr * (
            reward + self.discount * max_q_value
        )
        q_values[last_move] = new_q_value
        self.q_matrix[board_state] = q_values

    def _exploit(self, empty_cells: np.ndarray, board_state: str) -> int:
        """Performs action choice based on a q matrix

        Args:
            empty_cells: list of empty cells on a tic tac toe board
            board_state: string representing a current state in a board
        Returns:
            move: index of cell
        """
        q_values = self.q_matrix[board_state]
        possible_movements = q_values[empty_cells]
        movement = np.argmax(possible_movements)
        move = empty_cells[movement]
        self.last_move = move
        self.last_board_state = board_state
        return move

    def _explore(self, empty_cells: np.ndarray, board_state: str) -> int:
        """Performs action choice based on random choice

        Args:
            empty_cells: list of empty cells on a tic tac toe board
            board_state: string representing a current state in a board
        Returns:
            move: index of cell
        """
        move = np.random.choice(empty_cells)
        self.last_move = move
        self.last_board_state = board_state
        return move

    def _init_q_matrix(self):
        """Creates q_matrix as default dict with default value
        being a zeros vector of size 9
        """
        self.q_matrix = defaultdict(lambda: np.zeros(9, dtype=float))

    def dump_q_matrix(self):
        """Dumps q matrix to a file.

        Args:
            filename: file to store q matrix
        """
        q_matrix = dict(self.q_matrix)
        with open(self.filename, "wb") as f:
            pickle.dump(q_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)


class ConstantEpsilonFunction:
    def __init__(self, start_value: float = 0.2):
        """Returns constant value of epsilon.

        Args:
            start_value: threshold value
        """
        self.start_value = start_value

    def get_epsilon(self) -> float:
        return self.start_value


class GrowthEpsilonFunction:
    def __init__(self, mult: float):
        """Controls exploration/exploitation trade-off.
        Changes epsilon value over learning process.
        The more iteration done the less random steps agent makes.
        Given the formula T = exp^{-i * mult}.
        i - iteration number
        mult - a multiplier that controls value growth

        Args:
            mult: a multiplier that controls value growth
        """
        self.mult = mult
        self.iter = 0

    def get_epsilon(self) -> float:
        """Returns T value given the formula T = exp^{-i * mult} where
        i is the iteration number and mult is a multiplier that controls
        value growth.

        Returns:
            epsilon: float value of threshold to control
        exploration/exploitation.
        """
        epsilon = 1 - np.exp(-self.iter * self.mult)
        self.iter += 1
        return epsilon
