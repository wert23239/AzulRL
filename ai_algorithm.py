from ai_util import *
from azul_util import *
import tree_searcher


def switch_players(player_id):
    return (player_id + 1) % 2


@memoize
def eval_azul(state, turn):
    return score_azul_state(state, turn)


@memoize
def get_next_moves_fn(state, turn):
    return get_possible_azul_moves(state, turn)


@memoize
def is_terminal_fn(state):
    return azul_is_done(state)


def alpha_beta_find_state_value(alpha, beta, state, depth, turn):
    """
    alpha_beta helper function: Return the alpha_beta value of a particular
    state, given a particular depth to estimate to
    """
    if depth == 0 or is_terminal_fn(state):
        return eval_azul(state)

    best_val = None
    for move, new_state in get_next_moves_fn(state):
        val = -1 * alpha_beta_find_state_value(-1 * beta, -1 * alpha,
                                               new_state, depth-1,
                                               switch_players(turn))
        if best_val is None or val > best_val:
            best_val = val
            if val > alpha:
                alpha = val
            if beta <= alpha:
                break

    return best_val


def alpha_beta_search(state, depth, turn):
    best_val = None
    for action, new_state in get_next_moves_fn(state):
        val = -1 * alpha_beta_find_state_value(NEG_INFINITY, INFINITY,
                                               new_state, depth-1,
                                               switch_players(turn))
        if best_val is None or val > best_val[0]:
            best_val = (val, action)

    return best_val[1]


def run_search_function(state, turn, timeout=10):
    """
    Run the specified search function "search_fn" to increasing depths
    until "time" has expired; then return the most recent available return value

    "search_fn" must take the following arguments:
    state -- the state to search
    depth -- the depth to estimate to
    turn -- the player ID who's turn it currently is
    """

    eval_t = ContinuousThread(timeout=timeout, target=alpha_beta_search,
                              kwargs={'state': state, 'turn': turn})

    eval_t.setDaemon(True)
    eval_t.start()

    eval_t.join(timeout)

    # Note that the thread may not actually be done eating CPU cycles yet;
    # Python doesn't allow threads to be killed meaningfully...
    return int(eval_t.get_most_recent_val())


class AIAlgorithm():
    def action(self, state, possible_actions, turn, _):
        run_search_function(state, turn)

    def save(self, example):
        pass

    def train(self):
        pass
