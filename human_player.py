from action import Action
import environment_state


class HumanPlayer:
    def __init__(self):
        pass

    def action(self, state, possible_actions, turn, _):
        mosaic_template = [[1, 2, 3, 4, 5],
                       [5, 1, 2, 3, 4],
                       [4, 5, 1, 2, 3],
                       [3, 4, 5, 1, 2],
                       [2, 3, 4, 5, 1]]
        print("------ Your Mosaic (Left) ----- Mosaic Template (Right): -----")
        for i, row in enumerate(state.mosaics[turn]):
            print(i, row, "\t\t\t", mosaic_template[i])
        print("----------------------- Your Triangle: -----------------------")
        for i, row in enumerate(state.triangles[turn]):
            print(i, row)
        print("------------------------ Your Floor: -------------------------")
        print(state.floors[turn])
        print()
        print()
        print("---- Opponent's Mosaic (Left) --- Mosaic Template (Right): ---")
        for i, row in enumerate(state.mosaics[(turn + 1) % 2]):
            print(i, row, "\t\t\t", mosaic_template[i])
        print("-------------------- Opponent's Triangle: --------------------")
        for i, row in enumerate(state.triangles[(turn + 1) % 2]):
            print(i, row)
        print("---------------------- Opponent's Floor: ---------------------")
        print(state.floors[(turn + 1) % 2])
        print()
        print()
        print("-------------------------- Circles: --------------------------")
        for i, circle in enumerate(state.circles):
            print(i, sorted(circle))
        print("-------------------------- Center: ---------------------------")
        print(sorted(state.center))
        print("\nWhich circle would you like to pull from,")
        print("Which color would you like to pull from it,")
        print("And on which line of your triangle will you  place the tiles?")
        print("Format answer as '<circle>,<color>,<row>'.")
        print("Refer to the center as circle 5,")
        user_action_str = input(
            "And refer to the floor of your board as row 5:\n\n")
        while True:
            user_actions = user_action_str.split(",")
            if len(user_actions) == 3:
                # The following will crash the program if the user's input isn't
                # valid as integers.
                my_action = Action(
                    int(user_actions[0]), int(user_actions[1]), int(user_actions[2]))
                if my_action in possible_actions:
                    return (my_action,0,-1)
            user_action_str = input("Invalid action, try again:\n\n")

    def save(self, example):
        pass

    def train(self):
        pass

    def updateFinalReward(self,reward):
        pass