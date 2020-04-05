from environment import Environment
from DQN_agent import DQNAgent
from random_or_override import RandomOrOverride
from random_model import RandomModel
from human_player import HumanPlayer
from example import Example
import numpy as np
import sys

"""
Usage: python main.py player1_type player2_type <train_bots>
Acceptable types include 'random', 'bot', and 'human'.
Train bots is a boolean that will cause the model to train when it's set to
True and the 'bot' option is used.
"""

def score_to_reward(score):
    if score <= -2:
        return -1
    if score >= 2:
        return 1
    return 0

def assess_model(m1,random,e):
  m2 = RandomModel(random)
  player1_scores = []
  player1_wrong_guesses = []
  for games in range(500):
    state, turn, possible_actions = e.reset()
    done = False
    total_score = 0
    total_wrong_guesses = 0
    previous_turn = -1
    while not done:
      if turn == 0:
        player = m1
      else:
        player = m2
      previous_turn = turn
      action, action_num, num_wrong_guesses = player.action(state, possible_actions, turn, False)
      state, turn, possible_actions, score, done = e.move(action)
      if previous_turn==0:
        total_score += score
        if num_wrong_guesses != -1:
          total_wrong_guesses += num_wrong_guesses
        else:
          raise Exception
    player1_scores.append(total_score)
    if total_wrong_guesses == 0:
      print("total score", total_score)
    player1_wrong_guesses.append(total_wrong_guesses)
  print("accuracy", sum(player1_scores) / len(player1_scores))
  print("max score", max(player1_scores))
  print("average wrong guesses", sum(player1_wrong_guesses) / len(player1_wrong_guesses))
  print("min wrong guesses", min(player1_wrong_guesses))

def main(player1_type, player2_type, train_bots):
    random = RandomOrOverride()
    if player1_type == "bot":
        m1 = DQNAgent(random)
    elif player1_type == "random":
        m1 = RandomModel(random)
    else:
        m1 = HumanPlayer()
    if player2_type == "bot":
        m2 = DQNAgent(random)
    elif player2_type == "random":
        m2 = RandomModel(random)
    else:
        m2 = HumanPlayer()
    e = Environment(random)
    train_interval = 10
    accuracy_interval = 100

    number_of_games = 0
    while True:
        state, turn, possible_actions = e.reset()
        done = False
        previous_state = state  # FIX LATER
        while not done:
            if turn == 0:
                player = m1
            else:
                player = m2
            train = True
            if(type(m2) == HumanPlayer):
              train =False
            action, action_num, _ = player.action(state, possible_actions, turn, train)
            state, turn, possible_actions, score, done = e.move(action)
            if not train:
              print("score", score)
            reward = score_to_reward(score)
            example = Example(reward,action_num,possible_actions,previous_state.to_observable_state(),state.to_observable_state(),done)
            player.save(example)
            previous_state = state
        number_of_games+=1
        if number_of_games % train_interval == 0:
            m1.train()
            m2.train()
        if number_of_games % accuracy_interval == 0:
            assess_model(m1, random, e)

if __name__ == "__main__":
    player_types = ["random", "bot", "human"]
    valid_args = (
        len(sys.argv) >= 3
        and sys.argv[1] in player_types
        and sys.argv[2] in player_types
    ) or len(sys.argv) == 1
    if valid_args:
        train = True  # By default, we train the model.
        if len(sys.argv) == 1:
            main("bot", "bot", train)
        else:
          if len(sys.argv) >= 4:
            train = sys.argv[3]
          main(sys.argv[1], sys.argv[2], train)
    else:
        print("Usage: python main.py player1_type player2_type")
        print("Acceptable types include 'random', 'bot', and 'human'.")
