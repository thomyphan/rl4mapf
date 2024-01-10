from cactus.constants import *
from cactus.utils import assertContains
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_channels, nr_filters=128, kernel_size=3):
    return nn.Sequential(
            nn.Conv2d(in_channels, nr_filters, kernel_size, padding=2),
            nn.ReLU(),
            nn.Conv2d(nr_filters, nr_filters, kernel_size, padding=2),
            nn.ReLU(),
            nn.Conv2d(nr_filters, nr_filters, kernel_size, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(nr_filters, nr_filters, kernel_size, padding=2),
            nn.ReLU(),
            nn.Conv2d(nr_filters, nr_filters, kernel_size, padding=2),
            nn.ReLU(),
            nn.Conv2d(nr_filters, nr_filters, kernel_size, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(nr_filters, 512, 1),
            nn.ReLU()
        )

def seq_lstm_block(input_size, nr_hidden_dim=512):
    return nn.Sequential(
            nn.Linear(input_size, nr_hidden_dim),
            nn.ReLU(),
            nn.Linear(nr_hidden_dim, nr_hidden_dim),
            nn.ReLU()
        )

def output_head(input_size, output_size, nr_hidden_dim=512):
    return nn.Sequential(
            nn.Linear(input_size, nr_hidden_dim),
            nn.ReLU(),
            nn.Linear(nr_hidden_dim, output_size)
        )

class PRIMALModule(nn.Module):

    def __init__(self, params):
        assertContains(params, ENV_OBSERVATION_DIM)
        assertContains(params, HIDDEN_LAYER_DIM)
        assertContains(params, ENV_NR_ACTIONS)
        assertContains(params, ENV_NR_AGENTS)
        super(PRIMALModule, self).__init__()
        self.input_channels, self.input_size, _ = tuple(params[ENV_OBSERVATION_DIM])
        self.input_shape = self.input_channels*self.input_size*self.input_size
        self.hidden_layer_dim = params[HIDDEN_LAYER_DIM]
        self.nr_actions = params[ENV_NR_ACTIONS]
        self.nr_agents = params[ENV_NR_AGENTS]
        self.conv = conv_block(self.input_channels, nr_filters=128, kernel_size=3)
        self.intermediate_size = 512*6*6
        self.fc1 = nn.Linear(self.intermediate_size, 512)
        self.rnn = nn.LSTMCell(512, 512)
        self.policy_output = output_head(512, self.nr_actions)
        self.value_output = output_head(512, 1)

    def init_hidden_(self):
        return self.fc1.weight.new(1, 512).zero_()
    
    def init_hidden(self, batch_size):
        return self.init_hidden_().unsqueeze(0).expand(batch_size, self.nr_agents, -1)

    def forward(self, joint_observation, joint_hidden_state):
        joint_observation = joint_observation.view(-1, self.input_channels, self.input_size, self.input_size)
        x = self.conv(joint_observation)
        x = F.elu(self.fc1(x.view(-1, self.intermediate_size)))
        hidden_state = joint_hidden_state.reshape(-1, 512)
        next_hidden_state = self.rnn(x, hidden_state)
        action_logits = self.policy_output(next_hidden_state)
        values = self.value_output(next_hidden_state)
        return F.softmax(action_logits, dim=-1), values, next_hidden_state