from constants import PER_TURN, PER_GAME
from random_agent import RandomAgent


def assess_agent(m1, random, e, name, hyper_parameters):
    m2 = RandomAgent(random)
    player1_scores = []
    player1_rewards = []
    player1_wrong_guesses = []
    wins = 0
    losses = 0
    ties = 0
    for _ in range(hyper_parameters.assess_model_games):
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
            action = player.action(e, False)
            state, turn, _, score_delta, current_scores, done = e.move(action)
            if previous_turn == 0:
                # this may be one off by one because of how reward updates
                total_reward += score_to_reward(hyper_parameters.reward_function, score_delta, current_scores, done)
                total_score += score_delta
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
    result = str("player: {} avg_score: {} max_score:{} avg_reward: {} max_reward:{} ").format(
        name, avg_score, max_score, avg_reward, max_reward)
    print("win loss ratio against random: ",wins/(losses+wins+ties))
    print(result)
    print()
    return avg_score

def score_to_reward(reward_function, score_delta, current_scores, done):
    if(reward_function == PER_TURN):
        if score_delta < 0:
            return -1
        if score_delta > 0:
            return 1
        return 0
    elif(reward_function == PER_GAME):
        if(not done):
            return 0
        if(current_scores[0]>current_scores[1]):
            return 1
        if(current_scores[0]<current_scores[1]):
            return -1
        return 0
    raise Exception