from environment import Environment
from DQN_agent import DQNAgent
from random_or_override import RandomOrOverride

m1 = DQNAgent()
m2 = DQNAgent()
e = Environment(RandomOrOverride)

while True:
    state, turn, possible_actions = e.reset()
    done = False
    previous_action = None
    previous_state = None
    while not done:
        if turn == 1:
            player = m1
        else:
            player = m2
        a = player.action(state, possible_actions)
        state, turn, possible_actions, reward, done = e.move(a, player)
        player.save_temp(reward, a, state, previous_action, previous_state)
        previous_action = a
        previous_state = state
    player.save()

    # TODO(alexlambert): create test data