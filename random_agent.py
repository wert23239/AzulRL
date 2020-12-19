import random_or_override


class RandomAgent:
    def __init__(self, random_or_override):
        self.random_or_override = random_or_override

    def action(self, environment, _):
        l = list(environment.possible_actions)
        return self.random_or_override.random_sample(l, 1)[0]
