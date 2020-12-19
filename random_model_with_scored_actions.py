import random_or_override


class RandomModelWithScoredActions:
    def __init__(self, random_or_override):
        self.random_or_override = random_or_override

    def simulated_action(self, state, possible_actions):
        ret = {}
        sum_of_probs = 0
        for a in possible_actions:
            x = self.random_or_override.random_range_cont()
            sum_of_probs += x
            ret[a] = x
        normalized_ret = {}
        for a in ret:
            normalized_ret[a] = ret[a]/sum_of_probs
        return normalized_ret