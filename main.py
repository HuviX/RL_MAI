from agent import ConstantEpsilonFunction, QAgent, RandomAgent
from tictacenv import TicTacEnvironment


def main():
    q_agent_policy = ConstantEpsilonFunction(0.2)
    random_agent = RandomAgent(-1)
    env = TicTacEnvironment()
    q_agent = QAgent(1, "", q_agent_policy)
    agents = [random_agent, q_agent]
    for i in range(0, 5_000):
        agents = agents[::-1]
        state = env.reset()
        while not env.is_over:
            for p in agents:
                action = p.get_action(state)
                reward, state = env.step(action)
                p.set_reward(reward)
                if env.is_over:
                    break

        if i % 100:
            print(q_agent.reward, q_agent.amount_of_wins)
            print(random_agent.reward, random_agent.amount_of_wins)

    q_agent.dump_q_matrix("q_matrix.pkl")


if __name__ == "__main__":
    main()
