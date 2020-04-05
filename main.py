from environment import Environment
from DQN_agent import DQNAgent
from random_or_override import RandomOrOverride
from example import Example

def main():
    random = RandomOrOverride()
    m1 = DQNAgent(random)
    m2 = DQNAgent(random)
    e = Environment(random)

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
            action = player.action(state, possible_actions)
            state, turn, possible_actions, reward, done = e.move(action)
            example = Example(reward,action,state,previous_action,previous_state)
            player.save(example)
            previous_action = action
            previous_state = state
        print("round over")    

if __name__ == "__main__":
    main()