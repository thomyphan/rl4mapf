import cactus.algorithms as algorithms
import cactus.maps as maps
import cactus.experiments as experiments
from cactus.constants import *
from os.path import join
import sys

params = {}
completion_rates_final = []
filename = sys.argv[1]
map_size = sys.argv[2]
density = sys.argv[3]
for nr_agents in [4, 8, 16, 32, 64, 128, 256]:
    completion_rates = []
    for map_id in range(100):
        params[ENV_NR_AGENTS] = nr_agents
        params[MAP_NAME] = f"primal-{params[ENV_NR_AGENTS]}_agents_{map_size}_size_{density}_density_id_{map_id}_environment"
        params[EPISODES_PER_EPOCH] = 32
        params[ALGORITHM_NAME] = ALGORITHM_PPO_QMIX
        params[HIDDEN_LAYER_DIM] = 64
        params[NUMBER_OF_EPOCHS] = 1000
        params[EPISODES_PER_EPOCH] = 1
        params[EPOCH_LOG_INTERVAL] = 50
        params[ENV_TIME_LIMIT] = 256
        params[ENV_INIT_GOAL_RADIUS] = 10
        env = maps.make_test_map(params)
        controller = algorithms.make(params)
        controller.load_model_weights(join("output", filename))
        results = experiments.run_episodes(params[EPISODES_PER_EPOCH], [env], controller, params, training_mode=False, render_mode=False)
        completion_rates.append(results[COMPLETION_RATE])
        print(f"Run {nr_agents} agents in {map_id+1}/{100}: completion={numpy.mean(completion_rates)}\t\t\t\t", end='\r')
    completion_rates_final.append(numpy.mean(completion_rates))
print("{"+ f"completion_rate: {completion_rates_final}" + "}")
