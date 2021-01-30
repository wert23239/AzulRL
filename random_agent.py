import random_or_override


class RandomAgent:
    def __init__(self, random_or_override):
        self.name = "Random"
        self.random_or_override = random_or_override

    def action(self, environment):
        l = list(environment.possible_moves)
        return self.random_or_override.random_sample(l, 1)[0]
