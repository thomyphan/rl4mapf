import cactus.algorithms as algorithms
import cactus.maps as maps
import cactus.experiments as experiments
import cactus.data as data
import copy
from cactus.constants import *

nr_episodes = 1
params = {}
params[ENV_OBSERVATION_SIZE] = 7
params[ENV_NR_AGENTS] = 8
params[SAMPLE_NR_AGENTS] = params[ENV_NR_AGENTS]
params[HIDDEN_LAYER_DIM] = 64
params[NUMBER_OF_EPOCHS] = 5000
params[EPISODES_PER_EPOCH] = 32
params[EPOCH_LOG_INTERVAL] = 50
params[RADIUS_UPDATE_INTERVAL] = 250
params[ENV_TIME_LIMIT] = 100
params[TEST_INIT_GOAL_RADIUS] = None
params[ENV_GAMMA] = 1
params[RENDER_MODE] = False
params[ENV_MAKESPAN_MODE] = False
params[GRAD_NORM_CLIP] = 10
params[VDN_MODE] = False
params[REWARD_SHARING] = False
params[MIXING_HIDDEN_SIZE] = 128

def run(algorithm_name, curriculum_name):
    params[ALGORITHM_NAME] = algorithm_name
    params[CURRICULUM_NAME] = curriculum_name
    if "PRIMAL" in algorithm_name:
        params[ENV_USE_PRIMAL_REWARD] = True
        params[CURRICULUM_NAME] = RANDOM_CURRICULUM
    params[DIRECTORY] = f"output/{params[ENV_NR_AGENTS]}-agents_{curriculum_name}_{algorithm_name}"
    params[DIRECTORY] = data.mkdir_with_timestap(params[DIRECTORY])
    training_envs = maps.generate_training_maps(params)
    test_envs = [copy.deepcopy(e) for e in training_envs]
    controller = algorithms.make(params)
    results = experiments.run_training(training_envs, test_envs, controller, params)
    return controller, results

run(ALGORITHM_PPO_QMIX, CACTUS_CURRICULUM)
run(ALGORITHM_PPO_QMIX, RANDOM_CURRICULUM)
run(ALGORITHM_PPO_QPLEX, CACTUS_CURRICULUM)
run(ALGORITHM_PPO_QPLEX, RANDOM_CURRICULUM)
run(ALGORITHM_MAPPO, CACTUS_CURRICULUM)
run(ALGORITHM_MAPPO, RANDOM_CURRICULUM)
run(ALGORITHM_PRIMAL, RANDOM_CURRICULUM)