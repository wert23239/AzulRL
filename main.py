import sys

import numpy as np

from ai_algorithm import AIAlgorithm
from constants import PER_GAME
from DQN_agent import DQNAgent
from environment import Environment
from example import Example
from human_player import HumanPlayer
from hyper_parameters import HyperParameters
from random_model import RandomModel
from random_or_override import RandomOrOverride
from util import score_to_reward

from collections import defaultdict

"""
Usage: python main.py player1_type player2_type <train_bots>
Acceptable types include 'random', 'bot', and 'human'.
Train bots is a boolean that will cause the model to train when it's set to
True and the 'bot' option is used.
"""



def assess_model(m1, random, e, name, hyper_parameters):
    m2 = RandomModel(random)
    player1_scores = []
    player1_rewards = []
    player1_wrong_guesses = []
    for _ in range(500):
        state, turn, possible_actions = e.reset()
        done = False
        total_score = 0
        total_reward = 0
        total_wrong_guesses = 0
        previous_turn = -1
        while not done:
            if turn == 0:
                player = m1
            else:
                player = m2
            previous_turn = turn
            action, _, num_wrong_guesses = player.action(
                state.to_observable_state(turn), possible_actions, turn, False)
            state, turn, possible_actions, score, current_score, done = e.move(action)
            if previous_turn == 0:
                # this may be one off by one because of how reward updates
                total_reward += score_to_reward(hyper_parameters.reward_function,current_score,score,done)
                total_score += score
                if num_wrong_guesses != -1:
                    total_wrong_guesses += num_wrong_guesses
                else:
                    raise Exception
        player1_scores.append(total_score)
        player1_wrong_guesses.append(total_wrong_guesses)
        player1_rewards.append(total_reward)
    avg_score = sum(player1_scores) / len(player1_scores)
    max_score = max(player1_scores)
    avg_reward = sum(player1_rewards) / len(player1_rewards)
    max_reward = max(player1_rewards)
    average_wrong_guesses = sum(
        player1_wrong_guesses) / len(player1_wrong_guesses)
    min_wrong_guesses = min(player1_wrong_guesses)
    result = str("player: {} accuracy: {} max_score:{} avg_reward: {} max_reward:{}  average_wrong_guesses:{} min_wrong_guesses:{} ").format(
        name, avg_score, max_score, avg_reward, max_reward, average_wrong_guesses, min_wrong_guesses)
    print(result)
    print()
    return avg_score


def main(player1_type, player2_type, train_bots, hyper_parameters):
    random = RandomOrOverride()
    best_avg_score=0
    if player1_type == "bot":
        m1 = DQNAgent(random, hyper_parameters, player2_type == "human")
    elif player1_type == "random":
        m1 = RandomModel(random)
    elif player1_type == "human":
        m1 = HumanPlayer()
    else:
        m1 = AIAlgorithm()
    if player2_type == "bot":
        m2 = DQNAgent(random, hyper_parameters, player1_type == "human")
    elif player2_type == "random":
        m2 = RandomModel(random)
    elif player2_type == "human":
        m2 = HumanPlayer()
    else:
        m2 = AIAlgorithm()
    e = Environment(random)
    is_playing_bot = True
    if(type(m2) == HumanPlayer):
        is_playing_bot = False
    wins = 0
    losses = 0
    for number_of_games in range(1,hyper_parameters.max_games):
        state, turn, possible_actions = e.reset()
        done = False
        while not done:
            if turn == 0:
                player = m1
            else:
                player = m2
            action, _, _ = player.action(
                 state.to_observable_state(turn), possible_actions, turn, is_playing_bot)
            state, turn, possible_actions, score,current_score, done = e.move(action)
            if not is_playing_bot:
                print("score", score)
            reward = score_to_reward(hyper_parameters.reward_function,current_score,score,done)
            example = Example(reward, done)
            player.save(example)
            if(done):
                if(hyper_parameters.reward_function == PER_GAME):
                    m1.updateFinalReward(reward)
                    m2.updateFinalReward(reward*-1)
                if(current_score[0]>current_score[1]):
                    wins += 1
                else:
                    losses += 1
        if number_of_games % hyper_parameters.train_interval == 0 and is_playing_bot:
            m1.train()
            m2.train()
        if number_of_games % hyper_parameters.accuracy_interval == 0:
            print("win loss ratio: ",wins/(losses+wins))
            print("Epoch: ",number_of_games)
            avg_score=assess_model(m1, random, e, "player 1",hyper_parameters)
            best_avg_score = max(avg_score,best_avg_score)
            assess_model(m2, random, e, "player 2",hyper_parameters)
            wins = 0
            losses = 0
            if type(m1) == DQNAgent:
                total_first_choices = sum(
                  v for k, v in m1.first_choices.items())
                first_choice_list = [
                  (k, round(v*100/total_first_choices, 2))
                  for k, v in m1.first_choices.items()]
                print(sorted(first_choice_list, key=lambda x : x[1], reverse=True))
                m1.first_choices = defaultdict(int)
    return best_avg_score


if __name__ == "__main__":
    hyper_parmeters = HyperParameters()
    player_types = ["random", "bot", "human", "ai"]
    valid_args = (
        len(sys.argv) >= 3
        and sys.argv[1] in player_types
        and sys.argv[2] in player_types
    ) or len(sys.argv) == 1
    if valid_args:
        train = True  # By default, we train the model.
        if len(sys.argv) == 1:
            main("bot", "bot", train,hyper_parmeters)
        else:
            if len(sys.argv) >= 4:
                train = sys.argv[3]
            main(sys.argv[1], sys.argv[2], train, hyper_parmeters)
    else:
        print("Usage: python main.py player1_type player2_type")
        print("Acceptable types include 'random', 'bot', 'human', and 'ai'.")
