from collections import defaultdict
from typing import Tuple

from agent import ConstantEpsilonFunction, QAgent, RandomAgent
from tictacenv import TicTacEnvironment


def play_random_x_rounds(
    agents: Tuple[QAgent, RandomAgent],
    environment: TicTacEnvironment,
) -> None:
    players = defaultdict(lambda: 0)
    for _ in range(1000):
        agents = agents[::-1]
        state = environment.reset()
        is_over = False
        while not is_over:
            for p in agents:
                action = p.get_action(*state, 1.0)
                reward, state, is_over = environment.step(*action)
                if reward == 1:
                    players[p] += 1
                if is_over:
                    break
    print(players)
    return


def main():
    q_agent_policy = ConstantEpsilonFunction(0.1)
    random_agent = RandomAgent(-1)
    env = TicTacEnvironment()
    q_agent = QAgent(1, "", q_agent_policy)
    agents = [q_agent, random_agent]
    num_epochs = 100_000
    for _ in range(num_epochs):
        agents = agents[::-1]
        state = env.reset()
        is_over = False
        while not is_over:
            for p in agents:
                action = p.get_action(*state)
                reward, state, is_over = env.step(*action)
                p.set_reward(reward)
                if is_over:
                    break

    q_agent.dump_q_matrix("q_matrix.pkl")


if __name__ == "__main__":
    main()
