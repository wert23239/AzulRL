class Action:
  def __init__(self, circle, color, row):
    self.circle = circle
    self.color = color
    self.row = row

  def __hash__(self):
    return hash((self.circle, self.color, self.row))

  def __eq__(self, other):
    return self.circle == other.circle and self.color == other.color and self.row == other.row
  
  def __repr__(self):
    return "circle: {0}, color: {1}, row: {2}\t".format(self.circle, self.color, self.row)
