from numpy.core.arrayprint import BoolFormat
from agent import BaseAgent
from environment import BaseEnvironment


def main():
    env = BaseEnvironment([BaseAgent(1, "a") for _ in range(2)])
    env.print_board
    board_hash = env.get_board_hash()
    print(board_hash)

if __name__ == "__main__":
    main()