from agent import QAgent, ConstantEpsilonFunction, RandomAgent
from environment import Environment


def main():
    q_agent_policy = ConstantEpsilonFunction(0.2)
    random_agent = RandomAgent(-1)
    q_agent = QAgent(1, "", q_agent_policy)

    env = Environment([random_agent, q_agent])
    while not env.is_over:
        board_state = env.get_board_hash()
        empty_cells = env.get_empty_cells()
        # print(empty_cells)
        action_q = q_agent.get_action(empty_cells, board_state)
        # print("HERE", action_q)
        env.set_action(q_agent, action_q)
        print(env.board)
        empty_cells = env.get_empty_cells()
        action_r = random_agent.get_action(empty_cells)
        env.set_action(random_agent, action_r)
        print(env.board)

if __name__ == "__main__":
    main()