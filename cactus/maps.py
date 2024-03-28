from cactus.env.mapf_gridworld import MAPFGridWorld
from cactus.env.primal_gridworld import PRIMALGridWorld
from cactus.utils import get_param_or_default, assertEquals, assertContains
from cactus.constants import *
from os.path import join
import random
import cactus.env.env_generator as env_generator

DEFAULT_OBSTACLES = [
        [1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0]
    ]

def load_obstacles(filename):
    with open(filename, "r+") as file:
        lines = file.readlines()
        obstacles = []
        width = len(lines)
        for x, line in enumerate(lines[4:]):
            obstacle_line = []
            for y, character in enumerate(line.strip()):
                obstacle_line.append(character == "@")
            obstacles.append(obstacle_line)
            height = len(line.strip())
        return obstacles, width, height

def generate_training_maps(params):
    envs = []
    for size in [10, 10, 40, 80]:
        for density in [0, 0.1, 0.2, 0.3]:
            env,_ = env_generator.generate_mapf_gridworld(params[ENV_NR_AGENTS], size, density, params)
            envs.append(env)
    return envs

def make_test_map(params):
    assertContains(params, MAP_NAME)
    map_name = params[MAP_NAME]
    assert map_name.startswith("primal-"), f"Invalid test map name {map_name}"
    map_name = map_name.replace("primal-", "")
    filename = join("instances", "primal_test_envs", f"{map_name}.npy")
    params[ENV_PRIMAL_MAP] = numpy.load(filename)
    params[TORCH_DEVICE] = torch.device("cpu")
    return PRIMALGridWorld(params)

def make(params):
    map_name = get_param_or_default(params, MAP_NAME, None)
    if map_name is None and ENV_OBSTACLES not in params:
        params[ENV_OBSTACLES] = DEFAULT_OBSTACLES
        width = len(params[ENV_OBSTACLES])
        height = len(params[ENV_OBSTACLES][0])
        params[MAP_NAME] = "Default Map"
    elif map_name is not None and map_name.startswith("primal-"):
        map_name = map_name.replace("primal-", "")
        filename = join("instances", "primal_envs", f"{map_name}.npy")
        primal_map = numpy.load(filename)[0]
        primal_map[primal_map >= 0] = 0
        params[ENV_OBSTACLES] = -primal_map
        assertEquals(2, len(params[ENV_OBSTACLES].shape))
        params[ENV_TIME_LIMIT] = 256
        params[ENV_OBSERVATION_SIZE] = 7
        width = len(params[ENV_OBSTACLES])
        height = len(params[ENV_OBSTACLES][0])
    elif ENV_OBSTACLES in params:
        width = len(params[ENV_OBSTACLES])
        height = len(params[ENV_OBSTACLES][0])
        params[MAP_NAME] = "Custom Map"
    else:
        directory = get_param_or_default(params, INSTANCE_FOLDER, DEFAULT_FOLDER)
        filename = join(directory, f"{map_name}.map")
        params[ENV_OBSTACLES], width, height = load_obstacles(filename)
    max_map_size = max(width, height)
    params[ENV_TIME_LIMIT] = get_param_or_default(params, ENV_TIME_LIMIT, max_map_size*max_map_size)
    params[ENV_OBSERVATION_SIZE] = get_param_or_default(params, ENV_OBSERVATION_SIZE, 7)
    params[ENV_GAMMA] = get_param_or_default(params, ENV_GAMMA, 1)
    params[TORCH_DEVICE] = torch.device("cpu")
    env = MAPFGridWorld(params)
    params[ENV_NR_ACTIONS] = env.nr_actions
    return env