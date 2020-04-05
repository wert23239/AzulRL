import random_or_override


class RandomModel:
    def __init__(self, random_or_override):
        self.random_or_override = random_or_override

    def action(self, state, possible_actions):
        l = list(possible_actions)
        return self.random_or_override.random_sample(l, 1)[0]

    def save(self):
        pass
