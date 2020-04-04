class EnvironmentState:
  def __init__(self, tile_locations, mosaics, triangles, mosaic_bonuses,
               floors, one_piece, circles, center):
        self.tile_locations = tile_locations
        self.mosaics = mosaics
        self.triangles = triangles
        self.mosaic_bonuses = mosaic_bonuses
        self.floors = floors
        self.one_piece = one_piece
        self.circles = circles
        self.center = center
  
  def __hash__(self):
        return hash((self.tile_locations, self.mosaics, self.triangles,
                     self.mosaic_bonuses, self.floors, self.one_piece,
                     self.circles, self.center))

  def __eq__(self, other):
        return self.tile_locations == other.tile_locations and \
        self.mosaics == other.mosaics and \
        self.triangles == other.triangles and \
        self.mosaic_bonuses == other.mosaic_bonuses and \
        self.floors == other.floors and \
        self.one_piece == other.one_piece and \
        self.circles == other.circles and \
        self.center == other.center
  
  def __repr__(self):
        return "tile locations: {0}, mosaics: {1}, triangles: {2}, mosaic bonuses: {3}, floors: {4}, one_piece: {5}, circles: {6}, center: {7}\n".format(
            self.tile_locations, self.mosaics, self.triangles, self.mosaic_bonuses, self.floors, self.one_piece, self.circles, self.center
        )
