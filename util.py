from constants import PER_TURN, PER_GAME
from random_agent import RandomAgent
from ai_algorithm import AIAlgorithm

def assess_agent(m1, random, e, name, hyper_parameters,assess_count,player_metrics):
    m2 = AIAlgorithm()
    player1_scores = []
    player1_rewards = []
    player1_wrong_guesses = []
    wins = 0
    losses = 0
    ties = 0
    if  hasattr(m1.model, 'tensorboard'):
            m1.model.tensorboard.step = assess_count
    for i in range(hyper_ parameters.assess_model_games):
        turn = e.reset()
        done = False
        total_score = -10000000000
        total_reward = -100000000000
        previous_turn = -1
        while not done:
            if turn == 0:
                player = m1
            else:
                player = m2
            previous_turn = turn
            if(previous_turn == 0  and i==hyper_parameters.assess_model_games-1):
                player.action(e,False,True)
                break
            action = player.action(e, False)
            state, turn, _, score_delta, current_scores, done = e.move(action)
            if previous_turn == 0 or \
                (done and hyper_parameters.reward_function == PER_GAME) or \
                (done and turn == 0):
                total_reward += score_to_reward(hyper_parameters.reward_function, score_delta, current_scores, done)
            if done:
                total_score += current_scores[0] - current_scores[1]
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
    if hasattr(m1.model, 'tensorboard'):
        m1.model.tensorboard.update_stats(avg_score=avg_score,max_score=max_score,avg_reward=avg_reward, reward_max=max_reward,
    player_wins = player_metrics.wins,player_losses = player_metrics.losses, player_ties= player_metrics.ties,  illegal_moves =  player_metrics.illegal_moves, total_moves = player_metrics.total_moves)
    result = str("player: {} avg_score: {} max_score:{} avg_reward: {} max_reward:{} ").format(
        name, avg_score, max_score, avg_reward, max_reward)
    print("win loss ratio against random: ",wins/(losses+wins+ties))
    print(result)
    print()
    return avg_score

def score_to_reward(reward_function, score_delta, current_scores, done):
    # TODO: abandon all uses of score_delta
    # if(reward_function == PER_TURN):
    #     if score_delta < 0:
    #         return -1
    #     if score_delta > 0:
    #         return 1
    #     return 0
    if(reward_function == PER_GAME):
        if(not done):
            return 0
        if(current_scores[0]>current_scores[1]):
            return 1
        if(current_scores[0]<current_scores[1]):
            return -1
        return 0
    raise Exception