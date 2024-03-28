import numpy
import random
from cactus.utils import get_param_or_default
from cactus.constants import *
from cactus.env.mapf_gridworld import MAPFGridWorld

def generate_random_obstacles(size, density):
    obstacle_map = numpy.zeros((size, size))
    positions = [(x,y) for x in range(size) for y in range(size)]
    nr_obstacles = max(4, int(size*size*density) + 1)
    selected_positions = random.sample(positions, k=nr_obstacles)
    for x,y in selected_positions:
        if neighbor_locally_reachable((x,y), obstacle_map, size):
            obstacle_map[x][y] = 1
    return obstacle_map

def neighbor_locally_reachable(location, obstacle_map, size):
    x, y = location
    for dx, dy in [(-1,0), (0,-1),(1,0),(0,1)]:
        x_inbounds = x+dx > 0 and x+dx < size
        y_inbounds = y+dy > 0 and y+dy < size
        if x_inbounds and y_inbounds and locally_reachable(location, obstacle_map, size):
            return True
    return False

def locally_reachable(location, obstacle_map, size):
    x, y = location
    for dx, dy in [(-1,0), (0,-1),(1,0),(0,1)]:
        x_inbounds = x+dx > 0 and x+dx < size
        y_inbounds = y+dy > 0 and y+dy < size
        if x_inbounds and y_inbounds and not obstacle_map[x+dx,y+dy]:
            return True
    return False

def generate_mapf_gridworld(nr_agents, size, density, params):
    params[ENV_OBSTACLES] = generate_random_obstacles(size, density)
    params[ENV_TIME_LIMIT] = get_param_or_default(params, ENV_TIME_LIMIT, size*size)
    params[ENV_OBSERVATION_SIZE] = get_param_or_default(params, ENV_OBSERVATION_SIZE, 7)
    params[ENV_GAMMA] = get_param_or_default(params, ENV_GAMMA, 1)
    params[TORCH_DEVICE] = torch.device("cpu")
    params[ENV_NR_AGENTS] = nr_agents
    env = MAPFGridWorld(params)
    params[ENV_NR_ACTIONS] = env.nr_actions
    return env, params
