# AzulRL
A Reinforcement Learning Azul Bot and Environment

*Status*: Draft

*Created*: 2020-04-01

*Last updated*: 2020-04-04

![Azul](https://cdn.vox-cdn.com/thumbor/xuJINWQTYoxMVBK5G0tArv5U_KE=/0x0:600x600/2120x1413/filters:focal(252x252:348x348):format(webp)/cdn.vox-cdn.com/uploads/chorus_image/image/62724434/azulcover.0.jpg)

# SUMMARY 

The board game [Azul](https://en.wikipedia.org/wiki/Azul_(board_game)) is one that involves planning ahead and seamlessly making risk-benefit decisions. A player would benefit from an ability to make rapid probabilistic calculations and assessing the expected values of the various available options. So, a reinforcement learning algorithm is uniquely equipped to perform well.

Azul can be played by 2-4 players, but here we’ll focus on the 2-player version of the game for simplicity’s sake.

### Game Description / Terminology:

![Azul Board](https://s3.amazonaws.com/prod-media.gameinformer.com/styles/body_default/s3/2019/01/25/56c63ad8/05_0.jpg)

Each player has a **board**, and between them there are five **circles**. There are also colored **tiles** of five different colors: 20 tiles of each color. At the beginning of the game, all tiles are in the **bag** except for the **“1”** tile, which is placed in the **center** of the circles.

At the beginning of a **round**, four tiles are randomly chosen from the bag and placed on each of the five circles. Player 1 can choose to pick tiles from any of the five circles, taking all of a given color tile from that circle. The other tiles on the circle, which are not of that same color, are moved into the center. The tiles that were taken must be placed in one **row** of that player’s board’s **triangle**, starting at the rightmost position and going left. If the given row does not have space for all tiles to be placed, the overflow tiles must be placed at the **floor** of the player’s board, starting in the leftmost position and going right.

Then, on Player 2’s **turn**, they also choose tiles of a single color from one of the locations, except this time, the center is also a possible location (in addition to the circles). The first player to take tiles from the center must also take the “1” tile and place it at the floor of their board. This player will go first the following round.

Players 1 and 2 continue to take turns picking tiles, until there are no more tiles left on the four circles or in the center. At this point, the scores for the round are computed and the next round is prepared. 

Starting at the top of their triangle and moving down, for each completed row, each player will:



*   Move the rightmost tile into its rightful place (given color and row) on the **mosaic**.
*   Move the other tiles in that triangle row into the **top of the box**.
*   Add the new tile to their score:
    *   If there are no tiles immediately above/below or to the left/right of that tile, it’s +1 point. Otherwise, if there are tiles immediately above/below that tile, add the height of the continuous vertical tile line, and if there are tiles immediately to the left/right of that tile, add the width of the continuous horizontal tile line.
        *   In the example above, when the player on the left adds their red tile from their triangle to their mosaic, it will be worth 2 points. When they add their yellow tile, it will also be worth 2 points. When the player on the right adds their snowflake tile it will be worth 1 point, and when they add their blue tile it will also be worth 1 point.
    *   There are also bonuses: +10 points for having all five of a color on the mosaic, +7 for having a complete column, and +2 for having a complete row. Note that, once a player completes a row, the game is over.

The tiles occupying the floor of the board (including the “1” tile, if applicable) are worth negative points corresponding to the negative numbers above them. Each player subtracts their negative points from their score, and moves the tiles from the floor of their board to the top of the box.

The tiles in incomplete rows should stay where they are.

At the beginning of each round, four new tiles are dealt to each circle from the bag, and the “1” is placed in the center. If the bag becomes empty, it is refilled with the entire contents of the top of the box. The rounds continue until at least one player completes a row, at which point the player with the higher score wins.


# OBJECTIVE

Our goal is for the algorithm to be able to reliably beat a random-picker at the game (over 6,000 out of 10,000 games). If we are able to accomplish this, the next step (stretch goal) is to have the game play against humans: first Erica, then Alex, and then finally Richard. This will require the algorithm to be able to interface with other algorithms, as well as real, human users. It will need to be able to make a move, and then wait for a move from its opponent before proceeding. Another stretch goal is to implement a GUI that humans can use to play Azul against the model. The first version of this interface will likely just be through the terminal.


# OVERVIEW

There are three parts to this project. 



*   The Environment
*   Observation
*   The RL Agent

We need the environment to simulate the Azul game virtually. This allows a human to be able to play the bot, as well as the bot to play another bot. This will involve various levels of UI, configurable depending on the use case.

We’ll also need some way to give the bot the state of the game and other data. This is the “observation” the bot makes, or its understanding of the environment.

Finally, the RL agent itself will learn a policy for optimizing reward given its observations.


# DETAILED DESIGN


### The Environment

We will use the following modules:

**Utils that act on “EnvironmentState”**:



*   human\_state() // pretty print 
*   computer\_state() // returns ObservationState

**“Environment”**: 



*   Environment()
*   EnvironmentState state, int turn, set(Action) possible\_actions   reset()
*   EnvironmentState state, int turn, set(Action) possible\_actions, int score, bool done   move(Action action)
    *   Store player 1 last score and player 2 last score because score will need to take into account -1 \* the previous player’s score last turn

**“Main” pseudocode:**


```
m1 = BotModel()
if bot:
    m2 = BotModel()
elif human:
    m2 = HumanModel()
else:
    m2 = RandomModel()
e = Environment()

while True:
    turn = e.reset()
    done = False
    previous_action = None
    previous_state = None
    while not done:
        if turn == 1:
            player = m1
        else:
            player = m2
        a = player.action(state, possible_actions)
        state, history, turn, possible_actions, reward, done = e.move(a, player)
        player.save(reward, a, state, previous_action, previous_state)
        previous_action = a
        previous_state = state

```


**“Model”**:



*   HumanModel() (extends Model)
*   BotModel() (extends Model)
*   RandomModel() (extends Model)
*   Action action(EnvironmentState state)
    *   If human: this is when it asks for stdin
*   void save(int reward, Action action, EnvironmentState state, Action previous\_action, EnvironmentState state)
    *   If human: this is where it prints to console


### Observation

In order for the both to learn we model the environment into a trainable state using MDP’s. This means we need to understand steps, epochs, and states of the game. We will also need some way of defining accuracy. For now steps will be represented with every action you are able to make. This ends up being every other turn or so. Epochs will be represented with a single game of Azul. We will also bound a game to 7 rounds or less. This means there is a total of** 75 steps per epoch**.


#### Inputs

This table will explain how we will represent each input/action per game.


<table>
  <tr>
   <td><strong>Action</strong>
   </td>
   <td>Number per Step
   </td>
  </tr>
  <tr>
   <td>Which circle should we pick from?
   </td>
   <td>6 = (5 plus 1 for the center)
   </td>
  </tr>
  <tr>
   <td>Which color should we pick?
   </td>
   <td>5 options
   </td>
  </tr>
  <tr>
   <td>Which row should we put the tile?
   </td>
   <td>6 = (5 plus 1 for the floor)
   </td>
  </tr>
  <tr>
   <td><strong>Total actions</strong>
   </td>
   <td>180
   </td>
  </tr>
</table>


NOTE: we’ll define a common azul\_consts.py file that contains ALL constants


<table>
  <tr>
   <td><strong>Inputs (ObservationState)</strong>
   </td>
   <td>Amount of Values/Inputs per Step
   </td>
  </tr>
  <tr>
   <td>Tile Location Enum
<p>
inPlay, outOfPlay, outOfPlayTemp, inBag per Color Enum
   </td>
   <td>400V25I = 5(colors) * 5(inputs one per enum) * 20(values)
<p>
<code>Blue: [5 inPlay, 15 outOfPlay, 0 outOfPlayTemp, 0 inBag]</code>
<p>
<code>Red: … </code>
   </td>
  </tr>
  <tr>
   <td>Mosaics       
   </td>
   <td>250V50I  = 5(colors) * 25  -spot array * 2 (agent and opponent)
<p>
<code>[1,0,0,0,4,5...for 25] * 2</code>
   </td>
  </tr>
  <tr>
   <td>Triangles 
   </td>
   <td>250V10I = For each row: which color and how many
<p>
5(rows) * 5(colors) * 5(possible counts) * 2 (agent and opponent)
<p>
1 Red, 1 Blue, 1 Yellow, 3 White, 5 White
   </td>
  </tr>
  <tr>
   <td>Mosaics Bonuses 5 of a kind
   </td>
   <td>50V10I = 5(colors) * 5(number of tiles of the color placed on the mosaic) * 2(agent and opponent)
   </td>
  </tr>
  <tr>
   <td>Mosaics Bonuses Rows
   </td>
   <td>50V10I = 5(tiles) * 5(# rows) * 2(agent and opponent)
   </td>
  </tr>
  <tr>
   <td>Mosaics Bonuses Columns
   </td>
   <td>50V10I = 5(tiles) * 5(# columns) * 2(agent and opponent)
   </td>
  </tr>
  <tr>
   <td>“1” tile
   </td>
   <td>3V1I 0, 1, or 2 (which player has it, or unassigned)
   </td>
  </tr>
  <tr>
   <td>Circles
   </td>
   <td>100V20I = 4(spots) * 5(colors) * 5(circles)
   </td>
  </tr>
  <tr>
   <td>Center
   </td>
   <td>100V5I = 5(color) * 20(count per color)
   </td>
  </tr>
  <tr>
   <td>Total Values/Inputs
   </td>
   <td>1253V 136I
   </td>
  </tr>
</table>



### The RL Agent


#### The RL Method

For the initial run we are going to use the method DQN. In the future we may move towards policy gradients of **policy trees**.


#### The Reward Function

Each turn we will immediately find the score of a move and account for bonuses. Then we will subtract the score that the opponent made on the next turn. If you go last on a round then your score will not be subtracted. **If the end result is positive you gain +1. If it’s neutral you gain +0 and if it's negative you gain -1**.


#### Discount Factor

The actions you make in the game are very limited. There are max of 75 actions per epoch. However, each action can have an affect multiple turns later. This makes it hard to find a solid discount factor. Initially we will start at **.9**.


#### Steps for setup within a python3 venv (on MacOS):
1. Install python3, pip, wheel, and venv
2. `mkdir v_environment`
3. `python3 -m venv ../v_environment`
4. `source ../v_environment/bin/activate`
5. `pip install` ...
    * numpy
    * jinja2
    * keras
    * tensorflow
To leave the virual environment, simply run `deactivate`.
