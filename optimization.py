from bayes_opt import BayesianOptimization
from main import main as azul_main
from hyper_parameters import HyperParameters

def call_azul(batch_size,memory_length,discount_factor,epsilon_min,
    learning_rate,tau,train_interval,target_train_interval):
    hyper_parameters = HyperParameters(max_games=8000,batch_size=int(batch_size),memory_length=int(memory_length),
    discount_factor=discount_factor,epsion_min=epsilon_min, episilon_decay=995, learning_rate=learning_rate,
    tau=tau,train_interval=int(train_interval),accuracy_interval=100,target_train_interval=int(target_train_interval))
    return azul_main("bot","bot",True,hyper_parameters)

# Bounded region of parameter space
pbounds = {'batch_size': (16,128), 'memory_length': (1000,10000), 'discount_factor': (.75,.999), 'epsilon_min': (.01,.1),
'learning_rate': (.0025,.02),'tau': (.05,.2), 'train_interval': (10, 20),'target_train_interval': (1,5)}

optimizer = BayesianOptimization(
    f=call_azul,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=20,
    n_iter=100,
)

print(optimizer.max)
