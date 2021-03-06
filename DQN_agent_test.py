import unittest

import numpy as np

from action import Action
from DQN_agent import DQNAgent
from example import Example
from hyper_parameters import HyperParameters
from random_or_override import RandomOrOverride


DEFAULT_EXAMPLE_1 = Example(reward= 0,action= 0, possible_actions= [], state= [], next_state= [], done=True)
DEFAULT_EXAMPLE_2 = Example(reward= 5,action= 0, possible_actions= [], state= [], next_state= [], done=True)

class DQnAgentMethods(unittest.TestCase):

    def setUp(self):
        self.agent = DQNAgent(RandomOrOverride(), HyperParameters())

    def test_convert_action_num_and_convert_action(self):
        action = self.agent.convert_action_num(179)
        self.assertEqual(action,Action(5,5,5))
        action_num = self.agent.encode_action(action)
        self.assertEqual(action_num,179)

        action = self.agent.convert_action_num(30)
        self.assertEqual(action,Action(1,1,0))
        action_num = self.agent.encode_action(action)
        self.assertEqual(action_num,30)

        action = self.agent.convert_action_num(0)
        self.assertEqual(action,Action(0,1,0))
        action_num = self.agent.encode_action(action)
        self.assertEqual(action_num,0)

        action = self.agent.convert_action_num(1)
        self.assertEqual(action,Action(0,1,1))
        action_num = self.agent.encode_action(action)
        self.assertEqual(action_num,1)


        action = self.agent.convert_action_num(5)
        self.assertEqual(action,Action(0,1,5))
        action_num = self.agent.encode_action(action)
        self.assertEqual(action_num,5)

        action = self.agent.convert_action_num(6)
        self.assertEqual(action,Action(0,2,0))
        action_num = self.agent.encode_action(action)
        self.assertEqual(action_num,6)

    def test_convert_action_space_to_bit_mask_init(self):
        possible_actions = [Action(0,1,0),Action(0,1,1)]
        action_space = self.agent.convert_action_space_to_bit_mask(possible_actions)
        for i in range(len(action_space)):
          if(self.agent.encode_action(Action(0,1,0)) == i or
          (self.agent.encode_action(Action(0,1,1))) == i):
           self.assertEqual(action_space[i],1)
          else:
           self.assertEqual(action_space[i],0)

    def test_convert_action_space_to_bit_mask_random(self):
        possible_actions = [Action(2,1,2),Action(2,1,1)]
        action_space = self.agent.convert_action_space_to_bit_mask(possible_actions)
        for i in range(len(action_space)):
          if(self.agent.encode_action(Action(2,1,2)) == i or
          (self.agent.encode_action(Action(2,1,1))) == i):
           self.assertEqual(action_space[i],1)
          else:
           self.assertEqual(action_space[i],0)

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
    
    def test_updateFinalReward(self):
        self.agent.save(DEFAULT_EXAMPLE_2)
        self.agent.save(DEFAULT_EXAMPLE_1)
        self.assertEqual(self.agent.memory[-1].reward,0)
        self.agent.updateFinalReward(3)
        self.assertEqual(self.agent.memory[-1].reward,3)





if __name__ == '__main__':
    unittest.main()
