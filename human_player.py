from action import Action
import environment_state
import fileinput


class HumanPlayer:
    def __init__(self):
        pass

    def action(self, state, possible_actions, turn):
        print("------------------------ Your Mosaic: ------------------------")
        for row in state.mosaics[turn]:
            print(row)
        print("----------------------- Your Triangle: -----------------------")
        for row in state.triangles[turn]:
            print(row)
        print("------------------------ Your Floor: -------------------------")
        print(state.floors[turn])
        print("--------------------- Opponent's Mosaic: ---------------------")
        for row in state.mosaics[(turn + 1) % 2]:
            print(row)
        print("-------------------- Opponent's Triangle: --------------------")
        for row in state.triangles[(turn + 1) % 2]:
            print(row)
        print("---------------------- Opponent's Floor: ---------------------")
        print(state.floors[(turn + 1) % 2])
        print("-------------------------- Circles: --------------------------")
        for circle in state.circles:
            print(circle)
        print("-------------------------- Center: ---------------------------")
        print(state.center)
        print("Which circle would you like to pull from,")
        print("Which color would you like to pull from it,")
        print("And on which line of your triangle will you  place the tiles?")
        print("Format answer as '<circle>,<color>,<row>'.")
        print("Refer to the center as circle 5,")
        print("And refer to the floor of your board as row 5.")
        user_actions = fileinput.input().split(",")
        # The following will crash the program if the user's input isn't valid
        # as integers or if it doesn't contain the correct amount of integers.
        if len(user_actions) > 3:
            raise Exception
        return Action(
            int(user_actions[0]), int(user_actions[1]), int(user_actions[2]))

    def save(self, example):
        pass
