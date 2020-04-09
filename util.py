from constants import PER_TURN, PER_GAME
from main import score_to_reward

score_to_reward_functions(reward_function,hyper_parameters,current_scores,score,done):
    if(reward_function == PER_TURN):
        if score <= -2:
            return -1
        if score >= 2:
            return 1
        return 0
    else(reward_function == PER_GAME):
        if(not done):
            return 0
        if(current_score[0]>current_score[1]):
            return 1
        if(current_score[0]<current_score[1]):
            return -1
        return 0

