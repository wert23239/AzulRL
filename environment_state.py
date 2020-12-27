from constants import *
from collections import Counter
import numpy as np


class EnvironmentState:
    def __init__(self, tile_locations, mosaics, triangles, mosaic_bonuses,
                 floors, one_piece, circles, center):
        # Map from colors, to map from location to counts.
        self.tile_locations = tile_locations
        # List containing 2 5x5 arrays of colors.
        self.mosaics = mosaics
        # List containing 2 "5-triangle" arrays of colors
        # (ie [[0],[2,3],[1,1,2],[0,4,3,2],[4,4,1,2,2]])
        self.triangles = triangles
        # List containing 2 maps from bonus type to data. For five of a kind bonus, this is
        # a map from colors to counts. For others, this is a 5-item array of counts
        # corresponding to row/col index.
        self.mosaic_bonuses = mosaic_bonuses
        # List containing 2 7-item arrays of colors.
        self.floors = floors
        # Who is in posession of the going-first piece.
        self.one_piece = one_piece
        # 5x4 array of colors.
        self.circles = circles
        # array of colors.
        self.center = center

    def get_mosaics_in_order(self, player):
      return [self.mosaics[player], self.mosaics[(player + 1) % 2]]

    def get_triangles_in_order(self, player):
      return [self.triangles[player], self.triangles[(player + 1) % 2]]

    def get_mosaic_bonuses_in_order(self, player):
      return [self.mosaic_bonuses[player], self.mosaic_bonuses[(player + 1) % 2]]

    def get_floors_in_order(self, player):
      return [self.floors[player], self.floors[(player + 1) % 2]]

    def get_mirrored_one_piece(self, player):
      if self.one_piece == UNASSIGNED:
        return 2
      if self.one_piece == player:
        return 1
      return 0

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

    def to_observable_state(self, turn):
        # Handle the tile locations list first
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
        observable_state = []
        for l in tile_locations_list:
            observable_state += l

        # Next, the mosaics list and the triangles list.
        mosaics_list = [
            c for p in self.get_mosaics_in_order(turn) for r in p for c in r]
        triangles_list = [
            c for p in self.get_triangles_in_order(turn) for r in p for c in r]
        observable_state += mosaics_list + triangles_list


        # Next, the mosaic bonuses list.
        # mosaic_bonuses_list = [
        #     [p[FIVE_OF_A_KIND][c] for c in p[FIVE_OF_A_KIND]] +
        #     p[COLUMN_BONUS] + p[ROW_BONUS]
        #     for p in self.get_mosaic_bonuses_in_order(turn)]
        # for l in mosaic_bonuses_list:
        #     observable_state += l

        # Finally, the floors list, circles list, and center list.
        floors_list = [i for p in self.get_floors_in_order(turn) for i in p]
        circles_counters = [Counter(c) for c in self.circles]
        center_counter = Counter(self.center)
        circles_array = [[0 for a in range(5)] for b in range(5)]
        center_list = [0 for a in range(5)]
        for i, count in enumerate(circles_counters):
            for color in range(1, 6):
                circles_array[i][color-1] = circles_counters[i][color]
                center_list[color-1] = center_counter[color]
        circles_list = [column for row in circles_array for column in row]
        observable_state += \
            floors_list + [self.get_mirrored_one_piece(turn)] + circles_list + center_list
        return np.unpackbits(np.array(observable_state, dtype=np.uint8))

    def to_hashable_state(self, turn):
        return self.to_observable_state(turn).tostring()
