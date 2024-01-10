from cactus.constants import *
from cactus.utils import assertContains, assertEquals, get_param_or_default
import torch.nn as nn
import torch.nn.functional as F
import numpy

def preprocessing_module(nr_input_features, nr_hidden_units):
    return nn.Sequential(
            nn.Linear(nr_input_features, nr_hidden_units),
            nn.ELU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ELU()
        )

class QModule(nn.Module):

    def __init__(self, params):
        assertContains(params, ENV_OBSERVATION_DIM)
        assertContains(params, HIDDEN_LAYER_DIM)
        assertContains(params, ENV_NR_ACTIONS)
        assertContains(params, ENV_NR_AGENTS)
        super(QModule, self).__init__()
        self.input_shape = numpy.prod(params[ENV_OBSERVATION_DIM])
        self.hidden_layer_dim = params[HIDDEN_LAYER_DIM]
        self.nr_actions = params[ENV_NR_ACTIONS]
        self.nr_agents = params[ENV_NR_AGENTS]
        self.fc1 = preprocessing_module(self.input_shape, self.hidden_layer_dim)
        self.output = nn.Linear(self.hidden_layer_dim, self.nr_actions)

    def forward(self, joint_observation):
        joint_observation = joint_observation.view(-1, self.input_shape)
        x = self.fc1(joint_observation)
        return self.output(x)
    
class QMIXModule(nn.Module):
    def __init__(self, params):
        assertContains(params, ENV_OBSERVATION_DIM)
        assertContains(params, ENV_NR_AGENTS)
        assertContains(params, MIXING_HIDDEN_SIZE)
        super(QMIXModule, self).__init__()
        self.nr_agents = params[ENV_NR_AGENTS]
        self.mixing_hidden_size = params[MIXING_HIDDEN_SIZE]
        self.state_shape = numpy.prod(params[ENV_OBSERVATION_DIM])*self.nr_agents
        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_shape, self.mixing_hidden_size),
                                           nn.ELU(),
                                           nn.Linear(self.mixing_hidden_size, self.mixing_hidden_size * self.nr_agents))
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_shape, self.mixing_hidden_size),
                                           nn.ELU(),
                                           nn.Linear(self.mixing_hidden_size, self.mixing_hidden_size))
        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_shape, self.mixing_hidden_size)
        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_shape, self.mixing_hidden_size),
                               nn.ELU(),
                               nn.Linear(self.mixing_hidden_size, 1))

    def forward(self, global_state, Q_values):
        global_state = global_state.view(global_state.size(0), -1)
        w1 = torch.abs(self.hyper_w_1(global_state))
        b1 = self.hyper_b_1(global_state)
        w1 = w1.view(-1, self.nr_agents, self.mixing_hidden_size)
        b1 = b1.view(-1, 1, self.mixing_hidden_size)
        hidden = F.elu(torch.bmm(Q_values, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(global_state))
        w_final = w_final.view(-1, self.mixing_hidden_size, 1)
        # State-dependent bias
        v = self.V(global_state).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        return y.view(Q_values.size(0), -1, 1)

class QPLEXModule(nn.Module):
    def __init__(self, params):
        assertContains(params, ENV_OBSERVATION_DIM)
        assertContains(params, ENV_NR_AGENTS)
        assertContains(params, MIXING_HIDDEN_SIZE)
        assertContains(params, ENV_NR_ACTIONS)
        super(QPLEXModule, self).__init__()
        self.nr_heads = get_param_or_default(params, NR_ATTENTION_HEADS, 4)
        self.nr_agents = params[ENV_NR_AGENTS]
        self.nr_actions = params[ENV_NR_ACTIONS]
        self.mixing_hidden_size = params[MIXING_HIDDEN_SIZE]
        self.joint_action_dim = self.nr_actions*self.nr_agents
        self.state_shape = numpy.prod(params[ENV_OBSERVATION_DIM])*self.nr_agents
        self.hyper_weight = nn.Sequential(nn.Linear(self.state_shape, self.mixing_hidden_size),
                        nn.ELU(),
                        nn.Linear(self.mixing_hidden_size, 1))
        self.hyper_bias = nn.Sequential(nn.Linear(self.state_shape, self.mixing_hidden_size),
                        nn.ELU(),
                        nn.Linear(self.mixing_hidden_size, 1))
        self.lambdas = nn.ModuleList()
        self.phis = nn.ModuleList()
        self.vs = nn.ModuleList()
        for _ in range(self.nr_heads):
            self.lambdas.append(nn.Sequential(nn.Linear(self.state_shape + self.joint_action_dim, 32),
                                                    nn.ReLU(),
                                                    nn.Linear(32, 1)))
            self.phis.append(nn.Sequential(nn.Linear(self.state_shape, 32),
                                                    nn.ReLU(),
                                                    nn.Linear(32, 1)))
            self.vs.append(nn.Sequential(nn.Linear(self.state_shape, 32),
                                                    nn.ReLU(),
                                                    nn.Linear(32, 1)))

    def forward(self, V_i_local, A_i_local, observations, joint_actions):
        batch_size = observations.size(0)
        joint_observation = observations.view(batch_size, 1, self.state_shape)
        joint_observation = joint_observation.expand(batch_size, self.nr_agents, self.state_shape)
        joint_actions = F.one_hot(joint_actions.view(-1), self.nr_actions).view(batch_size, self.joint_action_dim)\
            .view(-1, 1, self.nr_agents*self.nr_actions)
        joint_actions = joint_actions.expand(batch_size, self.nr_agents, -1)
        w_i = self.hyper_weight(joint_observation).abs().view(-1, self.nr_agents) + EPSILON
        b_i = self.hyper_bias(joint_observation).view(-1, self.nr_agents)
        assertEquals(V_i_local.size(), w_i.size())
        assertEquals(A_i_local.size(), w_i.size())
        V_i = w_i*V_i_local + b_i
        V_tot = V_i.view(batch_size, self.nr_agents).sum(-1)
        A_i = w_i*A_i_local
        concat = torch.cat([joint_observation, joint_actions], dim=-1)
        all_Ks = [K(concat) for K in self.lambdas]
        all_Qs = [Q(joint_observation) for Q in self.phis]
        all_Vs = [V(joint_observation) for V in self.vs]
        lambdas = []
        for K, Q, V in zip(all_Ks, all_Qs, all_Vs):
            lambdas.append(F.sigmoid(K)*F.sigmoid(Q)*(V.abs() + EPSILON)) 

        lambdas = torch.stack(lambdas)
        lambdas = lambdas.sum(0).view(batch_size, self.nr_agents)
        A_tot = (lambdas*A_i).sum(-1)
        Q_tot = V_tot + A_tot
        return Q_tot.view(batch_size, -1, 1)
    
class CentralCriticModule(nn.Module):

    def __init__(self, params):
        assertContains(params, ENV_OBSERVATION_DIM)
        assertContains(params, MIXING_HIDDEN_SIZE)
        assertContains(params, ENV_NR_ACTIONS)
        assertContains(params, ENV_NR_AGENTS)
        super(CentralCriticModule, self).__init__()
        self.hidden_layer_dim = params[MIXING_HIDDEN_SIZE]
        self.nr_actions = params[ENV_NR_ACTIONS]
        self.nr_agents = params[ENV_NR_AGENTS]
        self.joint_action_dim = self.nr_actions*self.nr_agents
        self.observation_dim = numpy.prod(params[ENV_OBSERVATION_DIM])
        self.joint_observation_dim = self.observation_dim*self.nr_agents
        self.input_shape = self.joint_observation_dim + self.joint_action_dim
        self.fc1 = preprocessing_module(self.input_shape, self.hidden_layer_dim)
        self.output = nn.Linear(self.hidden_layer_dim, 1)

    def forward(self, joint_observation, joint_actions):
        joint_actions = F.one_hot(joint_actions.view(-1), self.nr_actions).view(-1, 1, self.nr_agents*self.nr_actions)
        joint_observation = joint_observation.view(-1, 1, self.joint_observation_dim)
        batch_size = joint_actions.size(0)
        assertEquals(batch_size, joint_observation.size(0))
        joint_observation = joint_observation.expand(batch_size, self.nr_agents, -1)
        observation_mask = torch.eye(self.nr_agents)\
            .view(1, self.nr_agents, self.nr_agents, 1)\
            .expand(batch_size, self.nr_agents, self.nr_agents, self.observation_dim) + 1.0
        joint_observation = joint_observation.view(batch_size, self.nr_agents, self.nr_agents, self.observation_dim)
        joint_observation = (joint_observation*observation_mask)\
            .view(batch_size, self.nr_agents, self.joint_observation_dim)
        joint_actions = joint_actions.expand(batch_size, self.nr_agents, -1)
        joint_actions = joint_actions.view(batch_size, self.nr_agents, self.nr_agents, self.nr_actions)
        action_mask = torch.eye(self.nr_agents)\
            .view(1, self.nr_agents, self.nr_agents, 1)\
            .expand(batch_size, self.nr_agents, self.nr_agents, self.nr_actions) + 1.0
        joint_actions = (joint_actions*action_mask)\
            .view(batch_size, self.nr_agents, self.joint_action_dim)
        x = self.fc1(torch.cat([joint_observation, joint_actions], dim=-1))
        return self.output(x).view(-1)