import unittest
from action import Action

import numpy as np

from DQN_agent import DQNAgent
from hyper_parameters import HyperParameters
from random_or_override import RandomOrOverride


class DQnAgentMethods(unittest.TestCase):

    def setUp(self):
        self.agent = DQNAgent(RandomOrOverride(), HyperParameters())
        pass

    def test_convert_action_num(self):
        action = self.agent.convert_action_num(179)
        self.assertEqual(action,Action(5,5,5))
        action = self.agent.convert_action_num(30)
        self.assertEqual(action,Action(1,1,0))
        action = self.agent.convert_action_num(0)
        self.assertEqual(action,Action(0,1,0))
        action = self.agent.convert_action_num(1)
        self.assertEqual(action,Action(0,1,1))
        action = self.agent.convert_action_num(1)
        self.assertEqual(action,Action(0,1,1))
        action = self.agent.convert_action_num(5)
        self.assertEqual(action,Action(0,1,5))
        action = self.agent.convert_action_num(6)
        self.assertEqual(action,Action(0,2,0))


    def test_get_best_possible_action_already_sorted(self):
        prediction_list = np.array([.1,.2,.3])
        possibleActions = [Action(0,1,0),Action(0,1,1),Action(0,1,2)]
        action,_,_ = self.agent.get_best_possible_action(prediction_list,possibleActions)
        self.assertEqual(action,Action(0,1,2))

    def test_get_best_possible_action_reverse_sorted(self):
        prediction_list = np.array([.3,.2,.1])
        possibleActions = [Action(0,1,0),Action(0,1,1),Action(0,1,2)]
        action,_,_ = self.agent.get_best_possible_action(prediction_list,possibleActions)
        self.assertEqual(action,Action(0,1,0))





if __name__ == '__main__': 
    unittest.main() 