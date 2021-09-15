import numpy as np
from random import randint 

class AbstractAgent:
    def __init__(self, sign, file_name):
        self.sign = sign
        self.file_name = file_name
    
    def get_action(self, env): 
        i, j = np.where(env == '-')
        rand = randint(0, len(i)-1)
        action = (i[rand], j[rand])
        return action
        
    def reward(self, reward, state, done):
        # reward - 0 - не выиграл, 1- выиграл
        # done - 0 - игра продолжается, 1 - игра окончена
        
        pass


class AbstractEnvironment:
    def __init__(self):
        self.environ = np.array([['-'] * 3 for _ in range(3)])
        
    def renew(self, sign, action):
        self.environ[action] = sign
        return self.condition(sign, self.environ)
        
    def condition(self, sign, environ):
        condition = 0 
        if (np.diagonal(environ) == sign).all() or (np.diagonal(environ[::-1]) == sign).all():
            condition = 1
        if (np.array([(x == sign).all() for x in environ])).any() or \
            (np.array([(x == sign).all() for x in environ.T])).any():
            condition = 1
        if '-' not in environ:
            condition = 2
        return condition
    
    def information(self, condition, sign):
        print(self.environ)
        if condition == 1:
            return "Победил {}".format(sign)
        elif condition == 2:
            return "Ничья"
        else:
            return "Ходит {}".format('X' if sign == 'O' else 'O')


def play_game():
    agent = AbstractAgent(1, 'file1')
    env = AbstractEnvironment()
    
    sign = 'X'
    cond = 0
    while cond == 0:
        action = agent.get_action(env.environ)
        cond = env.renew(sign, action)
        print(env.information(cond, sign))
        sign = 'X' if sign == 'O' else 'O'

play_game()