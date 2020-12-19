import unittest
from constants import *
from collections import Counter
from environment import Environment
from environment_state import EnvironmentState
from random_or_override import RandomOrOverride
from action import Action

def all_possible_combinations(possible_circles, possible_colors, possible_rows):
    result = set()
    for circle in possible_circles:
        for color in possible_colors:
            for row in possible_rows:
                result.add(Action(circle, color, row))
    possible_moves_list = list(result)
    possible_moves_list.sort(key=lambda a : a.circle * 100 + a.color * 10 + a.row)
    return possible_moves_list

class EnvironmentMethods(unittest.TestCase):

    def setUp(self):
        random_override=[
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.all_numbers_on_same_circle_environment = Environment(random_override=random_override)

    def test_setup_all_numbers_equal(self):
        turn = self.all_numbers_on_same_circle_environment.reset()
        self.assertEqual(turn, 1)
        possible_moves_list = list(self.all_numbers_on_same_circle_environment.possible_moves)
        possible_moves_list.sort(key=lambda a : a.circle * 100 + a.color * 10 + a.row)
        self.assertEqual(possible_moves_list, all_possible_combinations(range(5), [1], range(6)))

    def test_row_filling_move_all_numbers_equal(self):
        self.all_numbers_on_same_circle_environment.reset()
        _, turn, possible_moves, net_reward, current_score, done = \
            self.all_numbers_on_same_circle_environment.move(Action(0, 1, 3))
        self.assertEqual(turn, 0)
        self.assertEqual(done, False)
        self.assertEqual(current_score, [0, 1])
        self.assertEqual(net_reward, 1)  # 1 - 0
        possible_moves_list = list(possible_moves)
        possible_moves_list.sort(key=lambda a : a.circle * 100 + a.color * 10 + a.row)
        self.assertEqual(possible_moves_list,
                         all_possible_combinations(range(1, 5), [1], range(6)))
    
    def test_non_row_filling_move_all_numbers_equal(self):
        self.all_numbers_on_same_circle_environment.reset()
        self.all_numbers_on_same_circle_environment.move(Action(0, 1, 3))
        _, turn, possible_moves, net_reward, current_score, done = \
            self.all_numbers_on_same_circle_environment.move(Action(4, 1, 4))
        self.assertEqual(turn, 1)
        self.assertEqual(done, False)
        self.assertEqual(current_score, [0, 1])
        self.assertEqual(net_reward, -1)   # 0 - 1
        possible_moves_list = list(possible_moves)
        possible_moves_list.sort(key=lambda a : a.circle * 100 + a.color * 10 + a.row)
        self.assertEqual(possible_moves_list,
                         all_possible_combinations(range(1, 4), [1], [0, 1, 2, 4, 5]))
    
    def test_floor_move_all_numbers_equal(self):
        self.all_numbers_on_same_circle_environment.reset()
        self.all_numbers_on_same_circle_environment.move(Action(0, 1, 3))
        self.all_numbers_on_same_circle_environment.move(Action(4, 1, 4))
        _, turn, possible_moves, net_reward, current_score, done = \
            self.all_numbers_on_same_circle_environment.move(Action(3, 1, 5))
        self.assertEqual(turn, 0)
        self.assertEqual(done, False)
        self.assertEqual(current_score, [0, -5])
        self.assertEqual(net_reward, -6) # -6 - 0
        possible_moves_list = list(possible_moves)
        possible_moves_list.sort(key=lambda a : a.circle * 100 + a.color * 10 + a.row)
        self.assertEqual(possible_moves_list,
                         all_possible_combinations(range(1, 3), [1], range(6)))
    
    def test_row_finishing_move_all_numbers_equal(self):
        self.all_numbers_on_same_circle_environment.reset()
        self.all_numbers_on_same_circle_environment.move(Action(0, 1, 3))
        self.all_numbers_on_same_circle_environment.move(Action(4, 1, 4))
        self.all_numbers_on_same_circle_environment.move(Action(3, 1, 5))
        _, turn, possible_moves, net_reward, current_score, done = \
            self.all_numbers_on_same_circle_environment.move(Action(2, 1, 4))
        self.assertEqual(turn, 1)
        self.assertEqual(done, False)
        self.assertEqual(current_score, [-3, -5])
        self.assertEqual(net_reward, 3)  # 0 - -3
        possible_moves_list = list(possible_moves)
        possible_moves_list.sort(key=lambda a : a.circle * 100 + a.color * 10 + a.row)
        self.assertEqual(possible_moves_list,
                         all_possible_combinations([1], [1], [0, 1, 2, 4, 5]))
    
    def test_round_finishing_move_all_numbers_equal(self):
        self.all_numbers_on_same_circle_environment.reset()
        self.all_numbers_on_same_circle_environment.move(Action(0, 1, 3))
        self.all_numbers_on_same_circle_environment.move(Action(4, 1, 4))
        self.all_numbers_on_same_circle_environment.move(Action(3, 1, 5))
        self.all_numbers_on_same_circle_environment.move(Action(2, 1, 4))
        _, turn, possible_moves, net_reward, current_score,done = \
            self.all_numbers_on_same_circle_environment.move(Action(1, 1, 2))
        self.assertEqual(turn, 1)
        self.assertEqual(done, False)
        self.assertEqual(current_score, [-3, -6])
        self.assertEqual(net_reward, 2) # -1 - -3
        possible_moves_list = list(possible_moves)
        possible_moves_list.sort(key=lambda a : a.circle * 100 + a.color * 10 + a.row)
        self.assertEqual(possible_moves_list,
                         all_possible_combinations(range(5), [1], [0, 1, 4, 5]))
    
    def test_all_different_numbers_on_circle(self):
        random_override=[
            0, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
            1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        e = Environment(random_override=random_override)
        turn = e.reset()
        self.assertEqual(turn, 0)
        possible_moves_list = list(e.possible_moves)
        possible_moves_list.sort(key=lambda a : a.circle * 100 + a.color * 10 + a.row)
        self.assertEqual(possible_moves_list,
                         all_possible_combinations(range(5), range(1, 5), range(6)))

if __name__ == '__main__': 
    unittest.main() 
