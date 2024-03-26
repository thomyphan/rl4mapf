from cactus.tensorable import Tensorable
from cactus.utils import assertContains, get_param_or_default
from cactus.constants import *
import torch
import numpy

class Environment(Tensorable):

    def __init__(self, params) -> None:
        assertContains(params, ENV_TIME_LIMIT)
        assertContains(params, ENV_NR_AGENTS)
        assertContains(params, ENV_NR_ACTIONS)
        assertContains(params, ENV_OBSERVATION_DIM)
        assertContains(params, TORCH_DEVICE)
        self.device = params[TORCH_DEVICE]
        super(Environment, self).__init__(self.device)
        self.gamma = get_param_or_default(params, ENV_GAMMA, 1)
        self.observation_dim = params[ENV_OBSERVATION_DIM]
        self.nr_agents = params[ENV_NR_AGENTS]
        self.nr_actions = params[ENV_NR_ACTIONS]
        self.time_limit = params[ENV_TIME_LIMIT]
        self.agent_ids = self.as_int_tensor([i for i in range(self.nr_agents)])
        self.time_step = 0
        self.discounted_returns = self.float_zeros(self.nr_agents)
        self.undiscounted_returns = self.float_zeros(self.nr_agents)
        obs_dim = [self.nr_agents, numpy.prod(self.observation_dim)]
        self.joint_observation_buffer = self.float_zeros([self.time_limit+1] + obs_dim)

    def render(self):
        pass

    def is_terminated(self):
        pass

    def is_truncated(self):
        result = self.bool_zeros(self.nr_agents)
        result[:] = self.time_step >= self.time_limit
        return result

    def is_done(self):
        return torch.logical_or(self.is_truncated(), self.is_terminated())
    
    def is_done_all(self):
        return self.is_done().all()
    
    def joint_observation(self):
        return self.joint_observation_buffer[self.time_step]

    def reset(self):
        self.time_step = 0
        self.discounted_returns[:] = 0
        self.undiscounted_returns[:] = 0
        self.joint_observation_buffer[:] = 0
        return self.joint_observation_buffer
