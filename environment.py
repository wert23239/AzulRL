from constants import *
from collections import Counter
from environment_state import EnvironmentState
import random
from random_or_override import RandomOrOverride
from action import Action
from collections import deque
import numpy as np


class Environment:
    def __init__(self, round_limit=None, random_override=[], random_seed=None, history_size=7):
        self.random_override = random_override
        self.random_seed = random_seed
        self.round_limit = round_limit
        self.history_size = history_size
        self.history = deque(maxlen=history_size)

    def reset(self):
        for i in range(self.history_size):
            self.history.append(np.unpackbits(np.array([0] * 155, dtype=np.uint8)))
        self.random_or_override = RandomOrOverride(override=self.random_override, seed=self.random_seed)
        self.turn = self.random_or_override.random_range(0, 1)
        self.previous_rewards = [0, 0]  # players 1 and 2 both have scores of 0
        self.total_rewards = [0,0]
        self.done = False
        self.has_full_row = False
        self.round_count = 0

        # First, all tiles start in the bag.
        tile_locations = {
            color: {
                IN_PLAY: 0, OUT_OF_PLAY: 0, OUT_OF_PLAY_TEMP: 0, IN_BOX: 0, IN_BAG: 20
            } for color in [COLOR_A, COLOR_B, COLOR_C, COLOR_D, COLOR_E]
        }

        # All mosaics, triangles, and mosaic bonuses start empty.
        empty_mosaic_1 = [[NO_COLOR for j in range(5)] for i in range(5)]
        empty_mosaic_2 = [[NO_COLOR for j in range(5)] for i in range(5)]
        empty_triangle_1 = [[NO_COLOR for j in range(i+1)] for i in range(5)]
        empty_triangle_2 = [[NO_COLOR for j in range(i+1)] for i in range(5)]
        empty_mosaic_bonuses_1 = {
            FIVE_OF_A_KIND: {
                color: 0 for color in [COLOR_A, COLOR_B, COLOR_C, COLOR_D, COLOR_E]},
            COLUMN_BONUS: [0 for i in range(5)],
            ROW_BONUS: [0 for i in range(5)],
        }
        empty_mosaic_bonuses_2 = {
            FIVE_OF_A_KIND: {
                color: 0 for color in [COLOR_A, COLOR_B, COLOR_C, COLOR_D, COLOR_E]},
            COLUMN_BONUS: [0 for i in range(5)],
            ROW_BONUS: [0 for i in range(5)],
        }
        empty_floors_1 = [NO_COLOR for i in range(7)]
        empty_floors_2 = [NO_COLOR for i in range(7)]

        self.state = EnvironmentState(
            tile_locations,
            [empty_mosaic_1, empty_mosaic_2],  # mosaics
            [empty_triangle_1, empty_triangle_2],  # triangles
            [empty_mosaic_bonuses_1, empty_mosaic_bonuses_2],  # mosaic bonuses
            [empty_floors_1, empty_floors_2],  # floors
            UNASSIGNED,  # one_piece
            [[], [], [], [], []],  # circles
            []  # center
        )
        self.deal_tiles()
        self.find_possible_moves()
        self.history.append(self.state.to_observable_state(self.turn))
        return self.turn

    def move(self, action):
        reward = 0
        if action not in set(self.possible_moves):
            print("wtf")
            action = random.sample(self.possible_moves, 1)[0]
            reward -= 1
        # If choosing from the center, we may need to pay a penalty. Either way, get
        # the number of tiles chosen and remove them from the center.
        if action.circle == 5:
            if self.state.one_piece == UNASSIGNED:
                self.state.one_piece = self.turn
                reward += self.add_tiles_to_floor(ONE_TILE, 1)
            tile_counter = Counter(self.state.center)
            num_tiles = tile_counter[action.color]
            self.remove_from_center(action.color)
        # Get the number of tiles chosen and dump the others into the center.
        else:
            tile_counter = Counter(self.state.circles[action.circle])
            num_tiles = tile_counter[action.color]
            self.circle_remains_to_center(
                action.color, action.circle, tile_counter)

        # Add tiles to floor or row.
        if action.row == 5:
            reward += self.add_tiles_to_floor(action.color, num_tiles)
        else:
            reward += self.add_tiles_to_row(action.color,
                                            num_tiles, action.row)
        # Calculate reward and prepare for next turn.
        self.previous_rewards[self.turn] = reward
        self.total_rewards[self.turn] += reward
        net_reward = reward - self.previous_rewards[(self.turn + 1) % 2]
        if self.end_of_round():
            self.turn = self.state.one_piece
            self.state.one_piece = UNASSIGNED
            # This would only happen in the very rare case where nothing ever goes into the center.
            if self.turn == UNASSIGNED:
                self.turn = self.random_or_override.random_range(0, 1)
            self.prepare_next_round()
            self.round_count += 1
            self.done = self.has_full_row or (self.round_limit is not None and self.round_count == self.round_limit)
        else:
            self.turn = (self.turn + 1) % 2
        self.find_possible_moves()
        self.history.append(self.state.to_observable_state(self.turn))
        return self.state, self.history, self.turn, set(self.possible_moves), net_reward, self.total_rewards, self.done

    def prepare_next_round(self):
        # Put tiles from both players' filled rows into the box.
        for p in self.state.triangles:
            for row in p:
                filled_row = True
                for c in row:
                    if c == NO_COLOR:
                        filled_row = False
                        break
                if filled_row:
                    for i, color in enumerate(row):
                        row[i] = NO_COLOR
                        self.state.tile_locations[color][OUT_OF_PLAY_TEMP] -= 1
                        self.state.tile_locations[color][IN_BOX] += 1
        # Put tiles from both players' floors into the box.
        for p in self.state.floors:
            for i, color in enumerate(p):
                if color != NO_COLOR and color != ONE_TILE:
                    self.state.tile_locations[color][OUT_OF_PLAY_TEMP] -= 1
                    self.state.tile_locations[color][IN_BOX] += 1
                p[i] = NO_COLOR
        self.deal_tiles()

    def end_of_round(self):
        for c in self.state.circles:
            if len(c) != 0:
                return False
        return len(self.state.center) == 0

    def remove_from_center(self, color):
        self.state.center = list(filter((color).__ne__, self.state.center))

    def circle_remains_to_center(self, color, circle, tile_counter):
        self.state.circles[circle] = list(filter((color).__ne__,
                                                 self.state.circles[circle]))
        self.state.center += self.state.circles[circle]
        self.state.circles[circle] = []

    def add_tiles_to_floor(self, color, num_tiles):
        floor_scores = [-1, -1, -2, -2, -2, -3, -3]
        floor = self.state.floors[self.turn]
        reward = 0
        for i, c in enumerate(floor):
            if c == NO_COLOR:
                num_tiles -= 1
                floor[i] = color
                if color != ONE_TILE:
                    self.state.tile_locations[color][IN_PLAY] -= 1
                    self.state.tile_locations[color][OUT_OF_PLAY_TEMP] += 1
                reward += floor_scores[i]
            if num_tiles == 0:
                break
        # If we run out of space on the floor of the board, the remaining tiles
        # go in the box.
        if color != ONE_TILE:
            for i in range(num_tiles):
                self.state.tile_locations[color][IN_PLAY] -= 1
                self.state.tile_locations[color][IN_BOX] += 1

        return reward

    def add_tiles_to_row(self, color, num_tiles, row):
        reward = 0
        destination = self.state.triangles[self.turn][row]
        if not self.color_will_fit(color, destination):
            print("User tried to do an illegal action")
            print("color: ", color, " num_tiles: ", num_tiles, " row: ", row)
            raise Exception
        filled_row = True
        for i, tile in enumerate(destination):
            if tile == NO_COLOR:
                if num_tiles == 0:
                    filled_row = False
                    break
                destination[i] = color
                self.state.tile_locations[color][IN_PLAY] -= 1
                self.state.tile_locations[color][OUT_OF_PLAY_TEMP] += 1
                num_tiles -= 1
        if filled_row:
            reward += self.add_to_mosaic_reward(color, row)
            if num_tiles > 0:
                reward += self.add_tiles_to_floor(color, num_tiles)
        return reward

    def add_to_mosaic_reward(self, color, row):
        reward = 0
        color_order = [[COLOR_A, COLOR_B, COLOR_C, COLOR_D, COLOR_E],
                       [COLOR_E, COLOR_A, COLOR_B, COLOR_C, COLOR_D],
                       [COLOR_D, COLOR_E, COLOR_A, COLOR_B, COLOR_C],
                       [COLOR_C, COLOR_D, COLOR_E, COLOR_A, COLOR_B],
                       [COLOR_B, COLOR_C, COLOR_D, COLOR_E, COLOR_A]]
        col = color_order[row].index(color)
        self.state.mosaics[self.turn][row][col] = color
        vertical_neighbors, horizontal_neighbors = \
            self.vertical_and_horizontal_neighbors(self.state.mosaics[self.turn],
                                                   row, col)
        if vertical_neighbors == 1 and horizontal_neighbors == 1:
            reward += 1
        elif vertical_neighbors == 1:
            reward += horizontal_neighbors
        elif horizontal_neighbors == 1:
            reward += vertical_neighbors
        else:
            reward += vertical_neighbors + horizontal_neighbors

        self.state.mosaic_bonuses[self.turn][FIVE_OF_A_KIND][color] += 1
        if self.state.mosaic_bonuses[self.turn][FIVE_OF_A_KIND][color] == 5:
            reward += 10

        self.state.mosaic_bonuses[self.turn][COLUMN_BONUS][col] += 1
        if self.state.mosaic_bonuses[self.turn][COLUMN_BONUS][col] == 5:
            reward += 7

        self.state.mosaic_bonuses[self.turn][ROW_BONUS][row] += 1
        if self.state.mosaic_bonuses[self.turn][ROW_BONUS][row] == 5:
            reward += 2
            self.has_full_row = True
        return reward

    def vertical_and_horizontal_neighbors(self, grid, row, col):
        vertical_neighbors = 0
        horizontal_neighbors = 0
        original_row = row
        original_col = col
        # Look up
        while row >= 0 and grid[row][col] != NO_COLOR:
            vertical_neighbors += 1
            row -= 1
        # Look down
        row = original_row + 1  # don't double-count original row
        while row < len(grid) and grid[row][col] != NO_COLOR:
            vertical_neighbors += 1
            row += 1
        # Look left
        while col >= 0 and grid[original_row][col] != NO_COLOR:
            horizontal_neighbors += 1
            col -= 1
        # Look right
        col = original_col + 1  # don't double-count original col
        while col < len(grid[original_row]) and grid[original_row][col] != NO_COLOR:
            horizontal_neighbors += 1
            col += 1
        return vertical_neighbors, horizontal_neighbors

    def deal_tiles(self):
        # Randomly select tiles from the bag and put them in play.
        bag = [color for color in self.state.tile_locations
               for i in range(self.state.tile_locations[color][IN_BAG])]
        tiles_chosen = self.random_or_override.random_sample(
            bag, min(20, len(bag)))
        for color in tiles_chosen:
            self.state.tile_locations[color][IN_BAG] -= 1
            self.state.tile_locations[color][IN_PLAY] += 1
        if len(tiles_chosen) < 20:  # We may need to refill the bag.
            tiles_chosen += self.choose_more_tiles(tiles_chosen)
        self.state.circles = [tiles_chosen[i*4:(i+1)*4] for i in range(5)]

    def choose_more_tiles(self, tiles_chosen):
        # Contents of bag should equal what contents of the box once did.
        for color in self.state.tile_locations:
            self.state.tile_locations[color][IN_BAG] = \
                self.state.tile_locations[color][IN_BOX]
            self.state.tile_locations[color][IN_BOX] = 0
        bag = [color for color in self.state.tile_locations
               for i in range(self.state.tile_locations[color][IN_BAG])]
        # Randomly choose the remaining tiles needed and put them in play.
        new_tiles_chosen = self.random_or_override.random_sample(
            bag, 20 - len(tiles_chosen))
        for color in new_tiles_chosen:
            self.state.tile_locations[color][IN_BAG] -= 1
            self.state.tile_locations[color][IN_PLAY] += 1
        return new_tiles_chosen

    def find_possible_moves(self):
        self.possible_moves = []
        for circle_id, circle in enumerate(self.state.circles):
            if len(circle) == 0:
                continue
            self.find_possible_moves_in_tileset(circle_id, circle)
        # Center's circle id is 5.
        self.find_possible_moves_in_tileset(5, self.state.center)

    def find_possible_moves_in_tileset(self, circle_id, circle):
        for tile in set(circle):
            for i, row in enumerate(self.state.triangles[self.turn]):
                if self.color_will_fit(tile, row) \
                        and tile not in self.state.mosaics[self.turn][i]:
                    self.possible_moves.append(Action(circle_id, tile, i))
            # Use row 5 to mean the floor of the board, which is always an option.
            self.possible_moves.append(Action(circle_id, tile, 5))

    def color_will_fit(self, color, row):
        num_tiles_in_row = 0
        for tile in row:
            if tile != NO_COLOR:
                if tile != color:
                    return False
                else:
                    num_tiles_in_row += 1
        if len(row) == num_tiles_in_row:
            return False
        return True
