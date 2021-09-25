from agent import ConstantEpsilonFunction, QAgent, RandomAgent
from tictacenv import TicTacEnvironment


def main():
    q_agent_policy = ConstantEpsilonFunction(0.1)
    random_agent = RandomAgent(-1)
    env = TicTacEnvironment()
    q_agent = QAgent(1, "", q_agent_policy)
    agents = [q_agent, random_agent]
    num_epochs = 100_000
    for i in range(num_epochs):
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

        # if i % 1000:
        #     print(q_agent.reward, q_agent.amount_of_wins)
        #     print(random_agent.reward, random_agent.amount_of_wins)

    print(f"Avg. Reward for q_agent: {q_agent.reward / num_epochs}")
    print(f"Avg. Reward for random_agent: {random_agent.reward / num_epochs}")
    q_agent.dump_q_matrix("q_matrix.pkl")


if __name__ == "__main__":
    main()
