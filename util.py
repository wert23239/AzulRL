from colorama import Back, Style

from ai_algorithm import AIAlgorithm
from constants import PER_GAME, PER_TURN
from random_agent import RandomAgent
from tree_search_agent import TreeSearchAgent


def human_print(turn,state):
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

def assess_agent(m1, m2, e, hyper_parameters, tb_interval=0, player_metrics=None, m_tensorboard=None):
    player1_scores = []
    player1_rewards = []
    player1_wrong_guesses = []
    wins = 0
    losses = 0
    ties = 0
    games_to_assess = hyper_parameters.assess_model_games
    if type(m2) == AIAlgorithm:
        games_to_assess = 1
    for i in range(games_to_assess):
        m1.clear()
        if type(m2) == TreeSearchAgent:
            m2.clear()
        turn = e.reset()
        done = False
        total_score = 0
        total_reward = 0
        previous_turn = -1
        while not done:
            if turn == 0:
                player = m1
            else:
                player = m2
            previous_turn = turn
            action = player.action(e)
            state, turn, _, score_delta, current_scores, done = e.move(action)
            if done:
                total_score += current_scores[0] - current_scores[1]
                total_reward += score_to_reward(current_scores, done)
        player1_scores.append(total_score)
        player1_rewards.append(total_reward)
        if (total_reward > 0):
            wins += 1
        elif (total_reward < 0):
            losses += 1
        else:
            ties += 1
    avg_score = sum(player1_scores) / len(player1_scores)
    max_score = max(player1_scores)
    avg_reward = sum(player1_rewards) / len(player1_rewards)
    max_reward = max(player1_rewards)
    if m_tensorboard is not None and hasattr(m_tensorboard.model, 'tensorboard'):
        m_tensorboard.model.tensorboard.step = tb_interval
        turn = e.reset()
        m_tensorboard.model.no_op_action(e.state,e.possible_moves,turn)
        m_tensorboard.model.tensorboard.update_stats(
            avg_score=avg_score,
            max_score=max_score,
            avg_reward=avg_reward,
            reward_max=max_reward,
            total_moves=player_metrics.total_moves)
    result = str("{} vs. {} \t losses: {} ties: {} wins: {}  avg_score: {} max_score:{} avg_reward: {} max_reward:{}").format(
        m1.name, m2.name, losses, ties, wins, avg_score, max_score, avg_reward, max_reward)
    if (avg_reward > 0):
        print(Back.GREEN + result + Style.RESET_ALL)
    else:
        print(Back.RED + result + Style.RESET_ALL)
    if type(m2) == AIAlgorithm:
        return max_score
    return wins+ties*.5

def score_to_reward(current_scores, done):
    if(not done):
        return 0
    if(current_scores[0]>current_scores[1]):
        return 1
    if(current_scores[0]<current_scores[1]):
        return -1
    return 0
