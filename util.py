from constants import PER_TURN, PER_GAME

def score_to_reward(reward_function, score_delta, current_scores, done):
    if(reward_function == PER_TURN):
        if score_delta <= -2:
            return -1
        if score_delta >= 2:
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
