import random_or_override


class RandomModelWithScoredActions:
    def __init__(self, random_or_override):
        self.random_or_override = random_or_override

    def action(self, state, possible_actions):
        ret = {}
        for a in possible_actions:
            ret[a] = self.random_or_override.random_range_cont()
        return ret

    def save(self, example):
        pass

    def train(self):
        pass

    def updateFinalReward(self,reward):
        pass
