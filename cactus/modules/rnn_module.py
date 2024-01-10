from cactus.constants import *
from cactus.utils import assertContains
import torch.nn as nn
import torch.nn.functional as F
import numpy

class RNNModule(nn.Module):

    def __init__(self, params):
        assertContains(params, ENV_OBSERVATION_DIM)
        assertContains(params, HIDDEN_LAYER_DIM)
        assertContains(params, ENV_NR_ACTIONS)
        assertContains(params, ENV_NR_AGENTS)
        super(RNNModule, self).__init__()
        self.input_shape = numpy.prod(params[ENV_OBSERVATION_DIM])
        self.hidden_layer_dim = params[HIDDEN_LAYER_DIM]
        self.nr_actions = params[ENV_NR_ACTIONS]
        self.nr_agents = params[ENV_NR_AGENTS]
        self.fc1 = nn.Linear(self.input_shape, self.hidden_layer_dim)
        self.rnn = nn.GRUCell(self.hidden_layer_dim, self.hidden_layer_dim)
        self.action_head = nn.Linear(self.hidden_layer_dim, self.nr_actions)
        self.value_head = nn.Linear(self.hidden_layer_dim, 1)

    def init_hidden_(self):
        return self.fc1.weight.new(1, self.hidden_layer_dim).zero_()
    
    def init_hidden(self, batch_size):
        return self.init_hidden_().unsqueeze(0).expand(batch_size, self.nr_agents, -1)

    def forward(self, joint_observation, joint_hidden_state):
        joint_observation = joint_observation.view(-1, self.input_shape)
        x = F.elu(self.fc1(joint_observation))
        hidden_state = joint_hidden_state.reshape(-1, self.hidden_layer_dim)
        next_hidden_state = self.rnn(x, hidden_state)
        action_logits = self.action_head(next_hidden_state)
        value = self.value_head(next_hidden_state)
        return F.softmax(action_logits, dim=-1), value, next_hidden_state