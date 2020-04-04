from collections import Counter

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
        return "tile locations: {0}, mosaics: {1}, triangles: {2}, mosaic_bonuses: {3}, floors: {4}, one_piece: {5}, circles: {6}, center: {7}\n".format(
            self.tile_locations, self.mosaics, self.triangles, self.mosaic_bonuses, self.floors, self.one_piece, self.circles, self.center
        )
  
  def numbers_list(self):
        tile_locations_list = [
         [
          color,
          self.tile_locations[color][IN_PLAY],
          self.tile_locations[color][OUT_OF_PLAY],
          self.tile_locations[color][OUT_OF_PLAY_TEMP],
          self.tile_locations[color][IN_BOX],
          self.tile_locations[color][IN_BAG]
         ]
         for color in self.tile_locations]
        mosaics_list = [c for p in self.mosaics for r in p for c in r]
        triangles_list = [c for p in self.triangles for r in p for c in r]
        mosaic_bonuses_list = [
          [p[FIVE_OF_A_KIND][c] for c in p[FIVE_OF_A_KIND]] +
          p[COLUMN_BONUS] + p[ROW_BONUS]
          for p in self.mosaic_bonuses]
        floors_list = [i for p in self.floors for i in p]
        circles_counters = [Counter(c) for c in self.circles]
        center_counter = Counter(self.center)
        circles_array = [[0 for a in range(5)] for b in range(5)]
        center_list = [0 for a in range(5)]
        for i, count in enumerate(circles_counters):
          for color in range(1,6):
            circles_array[i][color-1] = circles_counters[i][color]
            center_list[color-1] = center_counter[color]
        circles_list = [column for row in circles_array for column in row]
        numbers_list = []
        for l in tile_locations_list:
          numbers_list += l
        numbers_list += mosaics_list + triangles_list
        for l in mosaic_bonuses_list:
          numbers_list += l
        numbers_list += \
          floors_list + [self.one_piece] + circles_list + center_list
        return numbers_list
