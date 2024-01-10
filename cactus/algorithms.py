from cactus.controller.controller import Controller
from cactus.controller.primal_controller import PRIMALController
from cactus.controller.a2c_controller import A2CController
from cactus.controller.ppo_controller import PPOController
from cactus.utils import get_param_or_default
from cactus.constants import *

def make(params):
    algorithm_name = get_param_or_default(params, ALGORITHM_NAME, DEFAULT_ALGORITHM)
    if algorithm_name == ALGORITHM_RANDOM:
        return Controller(params)
    if algorithm_name == ALGORITHM_PRIMAL:
        return PRIMALController(params)
    if algorithm_name == ALGORITHM_A2C:
        return A2CController(params)
    if algorithm_name == ALGORITHM_A2C_VDN:
        params[CRITIC_NAME] = CRITIC_VDN
        return A2CController(params)
    if algorithm_name == ALGORITHM_A2C_QMIX:
        params[MIXING_HIDDEN_SIZE] = 64
        params[CRITIC_NAME] = CRITIC_QMIX
        return A2CController(params)
    if algorithm_name == ALGORITHM_A2C_QPLEX:
        params[MIXING_HIDDEN_SIZE] = 64
        params[CRITIC_NAME] = CRITIC_QPLEX
        return A2CController(params)
    if algorithm_name == ALGORITHM_PPO_VDN:
        params[CRITIC_NAME] = CRITIC_VDN
        return PPOController(params)
    if algorithm_name == ALGORITHM_PPO_QMIX:
        params[MIXING_HIDDEN_SIZE] = 64
        params[CRITIC_NAME] = CRITIC_QMIX
        return PPOController(params)
    if algorithm_name == ALGORITHM_PPO_QPLEX:
        params[MIXING_HIDDEN_SIZE] = 64
        params[CRITIC_NAME] = CRITIC_QPLEX
        return PPOController(params)
    if algorithm_name == ALGORITHM_PPO:
        return PPOController(params)
    if algorithm_name == ALGORITHM_MAPPO:
        params[MIXING_HIDDEN_SIZE] = 64
        params[CRITIC_NAME] = CRITIC_CENTRAL
        return PPOController(params)
    raise ValueError(f"Unknown algorithm: '{algorithm_name}'")