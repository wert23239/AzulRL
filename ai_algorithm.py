from ai_util import *
import copy
import random
from environment import Environment
from random_or_override import RandomOrOverride


def create_new_environment(state, possible_moves, turn):
    r = RandomOrOverride()
    e = Environment(r)
    e.state = copy.deepcopy(state)
    e.possible_moves = copy.deepcopy(possible_moves)
    e.turn = turn
    e.done = False
    e.previous_rewards = [0, 0]
    return e


def alpha_beta_find_state_value(alpha, beta, action, depth, environment, scores):
    """
    alpha_beta helper function: Return the alpha_beta value of a particular
    state, given a particular depth to estimate to
    """
    e = create_new_environment(
        environment.state, environment.possible_moves, environment.turn)
    state, turn, possible_moves, score, done = e.move(action)
    if depth == 0 or done:
        return sum(scores) + score

    best_val = None
    for action in possible_moves:
        val = -1 * alpha_beta_find_state_value(-1 * beta, -1 * alpha,
                                               action, depth-1, e,
                                               scores + [score])
        if best_val is None or val > best_val:
            best_val = val
            if val > alpha:
                alpha = val
            if beta <= alpha:
                break

    return best_val


def alpha_beta_search(depth, environment):
    best_val = None
    for action in environment.possible_moves:
        e = create_new_environment(
            environment.state, environment.possible_moves, environment.turn)
        scores = []
        val = -1 * alpha_beta_find_state_value(NEG_INFINITY, INFINITY,
                                               action, depth-1,
                                               e, scores)
        if best_val is None or val > best_val[0]:
            best_val = (val, action)
    return best_val[1]


def run_search_function(environment, timeout=30):
    """
    Run the specified search function "search_fn" to increasing depths
    until "time" has expired; then return the most recent available return value

    "search_fn" must take the following arguments:
    environment -- an Environment object to mimic the environment of the game
    depth -- the depth to estimate to
    """
    eval_t = ContinuousThread(timeout=timeout, target=alpha_beta_search,
                              kwargs={'environment': environment})

    eval_t.setDaemon(True)
    eval_t.start()

    eval_t.join(timeout)

    # Note that the thread may not actually be done eating CPU cycles yet;
    # Python doesn't allow threads to be killed meaningfully...
    return eval_t.get_most_recent_val()


class AIAlgorithm():
    def action(self, state, possible_actions, turn, _):
        self.e = create_new_environment(state, possible_actions, turn)
        return (run_search_function(self.e), 0, 0)

    def save(self, example):
        pass

    def train(self):
        pass
