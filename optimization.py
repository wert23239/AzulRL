import random

from bayes_opt import BayesianOptimization

from settings import Settings
from constants import PER_GAME, SCORE, WIN_LOSS
from hyper_parameters import HyperParameters
from main import main as azul_main


def call_azul(learning_rate,num_simulations,ers,num_hl, hl_size, eps):
    # seed_param = None
    # if(int(ers)>50):
    #     seed_param = int(ers)
    max_games =int(2000/int(1))
    hyper_parameters = HyperParameters(
        max_games=max_games,
        assess_model_games=20,
        learning_rate=round(learning_rate,5),
        accuracy_interval=100,
        pgr=SCORE,
        num_simulations=int(num_simulations),
        ers=int(ers),
        round_limit=1,
        num_hl=int(num_hl),
        hl_size=int(hl_size),
        eps=eps,
        history_size=7)
    settings=Settings()
    return azul_main("bot","bot",hyper_parameters,settings)

# Bounded region of parameter space
pbounds = {'learning_rate': (.0001,.01),
           'num_simulations':(20,50.9999999),'ers':(0,100),
           'num_hl':(1,5.999),'hl_size':(64,512),'eps':(1e-8,1e-6)}

optimizer = BayesianOptimization(
    f=call_azul,
    pbounds=pbounds,
    random_state=1,
    verbose=2,
)

optimizer.maximize(
    init_points=4,
    n_iter=15,
)

print(optimizer.max)
