import copy
from environment import Environment
from random_or_override import RandomOrOverride

# Define 'INFINITY' and 'NEG_INFINITY'
try:
    INFINITY = float("infinity")
    NEG_INFINITY = float("-infinity")
# Windows doesn't support 'float("infinity")'.
except ValueError:
    INFINITY = float(1e3000)       # However, '1e3000' will overflow and return
    NEG_INFINITY = float(-1e3000)  # the magic float Infinity value anyway.


def alpha_beta_find_state_value(alpha, beta, action, environment, depth):
    """
    alpha_beta helper function: Return the alpha_beta value of a particular environment
    """
    turn = environment.turn
    e = copy.deepcopy(environment)
    _, _, _, possible_actions, _, total_rewards, done = e.move(action)
    if done or depth == 0:
        return total_rewards[turn] - total_rewards[(turn + 1) % 2]

    best_val = None
    for a in possible_actions:
        val = -1 * alpha_beta_find_state_value(-1 * beta, -1 * alpha,
                                               a, e, depth - 1)
        if best_val is None or val > best_val:
            best_val = val
            if val > alpha:
                alpha = val
            if beta <= alpha:
                break

    return best_val


class AIAlgorithm:
    def __init__(self):
        self.name = "AI Algorithm"

    def action(self, environment, max_depth=1):
        best_val = None
        for action in environment.possible_moves:
            val = alpha_beta_find_state_value(NEG_INFINITY, INFINITY, action, environment, max_depth)
            if best_val is None or val > best_val[0]:
                best_val = (val, action)
        return best_val[1]
