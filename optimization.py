from bayes_opt import BayesianOptimization
from main import main as azul_main
from hyper_parameters import HyperParameters
import random 
from constants import WIN_LOSS, SCORE,PER_GAME

def call_azul(learning_rate,alpha,gamma,policy_gradient_reward,num_simulations,ers,
    round_limit, num_hl, hl_size,max_depth):
    policy_gradient_reward_param = WIN_LOSS
    if(policy_gradient_reward>.5):
        policy_gradient_reward_param = SCORE
    alpha_param = 1
    print(alpha)
    if(alpha<.5):
        alpha_param = 1e-7
    seed_param = None
    if(int(ers)>50):
        seed_param = int(ers)
    max_games =int(1500/int(round_limit))
    hyper_parameters = HyperParameters(max_games=max_games,assess_model_games=20,learning_rate=round(learning_rate,3),
    alpha=alpha_param,gamma=round(gamma,3), print_model_nn=False,accuracy_interval=100,save_interval=1000,
    reward_function=PER_GAME,pgr=policy_gradient_reward_param,num_simulations=int(num_simulations),
    ers=seed_param,round_limit=int(round_limit),num_hl=int(num_hl),
    hl_size=int(hl_size),max_depth=int(max_depth))
    return azul_main("bot","bot",hyper_parameters)

# Bounded region of parameter space
pbounds = {'learning_rate': (.0025,.01),'alpha': (0,1),'gamma': (.9,.99),'policy_gradient_reward':(0,1),
           'num_simulations':(1,7),'ers':(0,100),'round_limit':(1,2.99999),
           'num_hl':(1,3.999),'hl_size':(64,256),'max_depth':(1,7.999)}

optimizer = BayesianOptimization(
    f=call_azul,
    pbounds=pbounds,
    random_state=1,
    verbose=2,
)

optimizer.maximize(
    init_points=2,
    n_iter=10,
)

print(optimizer.max)
