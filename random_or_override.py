import random

"""
Returns either a random int, or an overriden value (for testing).
When overriding, pass in a sequence of ints for this function to return.
"""
class RandomOrOverride:
  def __init__(self, override=[]):
    self.override = override
    self.override_index = 0
    random.seed()

  def random_range(self, min, max):
    if self.override_index >= len(self.override):
      return random.randint(min, max)
    else:
      res = self.override[self.override_index]
      self.override_index += 1
      return res
  
  def random_range_cont(self, min, max):
    if self.override_index >= len(self.override):
      return random.random(min, max)
    else:
      res = self.override[self.override_index]
      self.override_index += 1
      return res


  def random_sample(self, population, k):
    if self.override_index + k > len(self.override):
      return random.sample(population, k)
    else:
      res = self.override[self.override_index:self.override_index + k]
      self.override_index += k
      return res
