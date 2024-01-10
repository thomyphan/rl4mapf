from cactus.constants import *
from cactus.utils import assertContains
import torch.nn as nn
import numpy

def preprocessing_module(nr_input_features, nr_hidden_units):
    return nn.Sequential(
            nn.Linear(nr_input_features, nr_hidden_units),
            nn.ELU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ELU()
        )

class FFNModule(nn.Module):

    def __init__(self, params):
        assertContains(params, ENV_OBSERVATION_DIM)
        assertContains(params, HIDDEN_LAYER_DIM)
        assertContains(params, ENV_NR_ACTIONS)
        assertContains(params, ENV_NR_AGENTS)
        super(FFNModule, self).__init__()
        self.input_shape = numpy.prod(params[ENV_OBSERVATION_DIM])
        self.hidden_layer_dim = params[HIDDEN_LAYER_DIM]
        self.nr_actions = params[ENV_NR_ACTIONS]
        self.nr_agents = params[ENV_NR_AGENTS]
        self.fc1 = preprocessing_module(self.input_shape, self.hidden_layer_dim)
        self.output = nn.Linear(self.hidden_layer_dim, params[OUTPUT_DIM])

    def forward(self, joint_observation):
        joint_observation = joint_observation.view(-1, self.input_shape)
        x = self.fc1(joint_observation)
        return self.output(x)