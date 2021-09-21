from agent import ConstantEpsilonFunction, QAgent, RandomAgent
from environment import Environment


def main():
    q_agent_policy = ConstantEpsilonFunction(0.2)
    random_agent = RandomAgent(-1)
    q_agent = QAgent(1, "", q_agent_policy)
    agents = [random_agent, q_agent][::-1]
    for i in range(0, 5_000):
        env = Environment()
        while not env.is_over:
            for p in agents:
                board_state = env.get_board_hash()
                empty_cells = env.get_empty_cells()
                action = p.get_action(empty_cells, board_state)
                env.set_action(p, action)
                if env.is_over:
                    break
        if i % 100:
            print(q_agent.reward, q_agent.amount_of_wins)
            print(random_agent.reward, random_agent.amount_of_wins)

    q_agent.dump_q_matrix("q_matrix.pkl")


if __name__ == "__main__":
    main()
