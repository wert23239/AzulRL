from constants import PER_TURN, PER_GAME
from random_agent import RandomAgent
from ai_algorithm import AIAlgorithm

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

def assess_agent(m1, m2, e, hyper_parameters,assess_count,player_metrics,add_to_tensorboard=False):
    player1_scores = []
    player1_rewards = []
    player1_wrong_guesses = []
    wins = 0
    losses = 0
    ties = 0
    if  hasattr(m1.model, 'tensorboard') and add_to_tensorboard:
            m1.model.tensorboard.step = assess_count
    games_to_assess = hyper_parameters.assess_model_games
    if type(m2) == AIAlgorithm:
        games_to_assess = 1
    for i in range(games_to_assess):
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
    if(hasattr(m1.model, 'tensorboard') and add_to_tensorboard):
        turn = e.reset()
        m1.model.no_op_action(e.state,e.possible_moves,turn)
    avg_score = sum(player1_scores) / len(player1_scores)
    max_score = max(player1_scores)
    avg_reward = sum(player1_rewards) / len(player1_rewards)
    max_reward = max(player1_rewards)
    if hasattr(m1.model, 'tensorboard') and add_to_tensorboard:
        m1.model.tensorboard.update_stats(avg_score=avg_score,max_score=max_score,avg_reward=avg_reward, reward_max=max_reward,
    player_wins = player_metrics.wins,player_losses = player_metrics.losses, player_ties= player_metrics.ties,  illegal_moves =  player_metrics.illegal_moves, total_moves = player_metrics.total_moves)
    result = str("avg_score: {} max_score:{} avg_reward: {} max_reward:{} ").format(
        avg_score, max_score, avg_reward, max_reward)
    print(result)
    if type(m2) == AIAlgorithm:
        return max_score
    return wins

def score_to_reward(current_scores, done):
    if(not done):
        return 0
    if(current_scores[0]>current_scores[1]):
        return 1
    if(current_scores[0]<current_scores[1]):
        return -1
    return 0