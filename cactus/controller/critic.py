from cactus.modules.q_module import QModule, QMIXModule, CentralCriticModule, QPLEXModule
from cactus.utils import assertEquals, get_param_or_default
from cactus.constants import *
from os.path import join
import torch
import torch.nn as nn
import torch.nn.functional as F

class CentralCritic:

    def __init__(self, params) -> None:
        self.device = params[TORCH_DEVICE]
        self.grad_norm_clip = get_param_or_default(params, GRAD_NORM_CLIP, 1)
        self.learning_rate = get_param_or_default(params, LEARNING_RATE, 0.001)
        self.q_net = CentralCriticModule(params)
        self.parameters = list(self.q_net.parameters())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

    def counterfactual_baseline(self, observations, actions, probs):
        return self.q_net(observations, actions).view(-1)

    def get_parameter_count(self):
        return sum(p.numel() for p in self.q_net.parameters() if p.requires_grad)
    
    def save_model_weights(self, path):
        critic_path = join(path, CRITIC_NET_FILENAME)
        torch.save(self.q_net.state_dict(), critic_path)

    def load_model_weights(self, path):
        critic_path = join(path, CRITIC_NET_FILENAME)
        self.q_net.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.q_net.eval()

    def train(self, observations, actions, targets):
        Q_values = self.q_net(observations, actions).view(-1)
        targets = targets.view(-1)
        assertEquals(Q_values.size(), targets.size())
        loss = F.mse_loss(Q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters, self.grad_norm_clip)
        self.optimizer.step()

class QCritic:

    def __init__(self, params) -> None:
        self.grad_norm_clip = get_param_or_default(params, GRAD_NORM_CLIP, 1)
        self.learning_rate = get_param_or_default(params, LEARNING_RATE, 0.001)
        self.q_net = QModule(params)
        self.device = params[TORCH_DEVICE]
        self.parameters = list(self.q_net.parameters())
        self.nr_agents = params[ENV_NR_AGENTS]
        self.nr_actions = params[ENV_NR_ACTIONS]

    def save_model_weights(self, path):
        critic_path = join(path, CRITIC_NET_FILENAME)
        torch.save(self.q_net.state_dict(), critic_path)

    def load_model_weights(self, path):
        critic_path = join(path, CRITIC_NET_FILENAME)
        self.q_net.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.q_net.eval()

    def get_parameter_count(self):
        return sum(p.numel() for p in self.q_net.parameters() if p.requires_grad)

    def get_local_value(self, observations):
        return self.q_net(observations)
    
    def counterfactual_baseline(self, observations, actions, probs):
        probs = probs.view(-1, self.nr_actions)
        local_Q_values = self.get_local_value(observations).view(-1, self.nr_actions)
        assertEquals(probs.size(), local_Q_values.size())
        return (probs*local_Q_values).sum(1)

    def global_value(self, observations):
        pass

    def train(self, observations, actions, targets):
        pass

class VDNCritic(QCritic):

    def __init__(self, params) -> None:
        super(VDNCritic, self).__init__(params)
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)

    def train(self, observations, actions, targets):
        new_batch_size = observations.size(0)
        batch_size = int(observations.size(0)/self.nr_agents)
        observations = observations.view(new_batch_size, -1)
        actions = actions.view(new_batch_size, 1)
        targets = targets.view(batch_size, self.nr_agents)
        local_Q_values = self.get_local_value(observations).gather(1, actions).squeeze()
        local_Q_values = local_Q_values.view(batch_size, self.nr_agents)
        Q_values = self.global_value(local_Q_values, observations)
        targets = targets.sum(-1)
        assertEquals(Q_values.size(), targets.size())
        loss = F.mse_loss(Q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters, self.grad_norm_clip)
        self.optimizer.step()

    def global_value(self, local_Q_values, observations):
        return local_Q_values.sum(-1)

class QMIXCritic(VDNCritic):

    def __init__(self, params) -> None:
        super(VDNCritic, self).__init__(params)
        self.mixing_network = QMIXModule(params)
        self.parameters += list(self.mixing_network.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)

    def save_model_weights(self, path):
        super(QMIXCritic, self).save_model_weights(path)
        critic_path = join(path, MIXER_NET_FILENAME)
        torch.save(self.mixing_network.state_dict(), critic_path)

    def load_model_weights(self, path):
        super(QMIXCritic, self).load_model_weights(path)
        critic_path = join(path, MIXER_NET_FILENAME)
        self.mixing_network.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.mixing_network.eval()

    def get_parameter_count(self):
        parameter_count = super(QMIXCritic, self).get_parameter_count()
        mixer_parameter_count = sum(p.numel() for p in self.mixing_network.parameters() if p.requires_grad)
        return parameter_count + mixer_parameter_count

    def global_value(self, local_Q_values, observations):
        batch_size = int(observations.size(0)/self.nr_agents)
        observations = observations.view(batch_size, self.nr_agents, -1)
        local_Q_values = local_Q_values.view(-1, 1, self.nr_agents)
        return self.mixing_network(observations, local_Q_values).squeeze()

class QPLEXCritic(QCritic):

    def __init__(self, params) -> None:
        super(QPLEXCritic, self).__init__(params)
        self.mixing_network = QPLEXModule(params)
        self.parameters += list(self.mixing_network.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)

    def train(self, observations, actions, targets):
        new_batch_size = observations.size(0)
        batch_size = int(observations.size(0)/self.nr_agents)
        observations = observations.view(new_batch_size, -1)
        actions = actions.view(new_batch_size, 1)
        targets = targets.view(batch_size, self.nr_agents)
        current_Q_values = self.get_local_value(observations)
        local_Q_values = current_Q_values.gather(1, actions).squeeze()
        local_Q_values = local_Q_values.view(batch_size, self.nr_agents)
        max_Q_values = current_Q_values.max(-1)[0]
        max_Q_values = max_Q_values.view(batch_size, self.nr_agents)
        Q_values = self.global_value(local_Q_values, max_Q_values, observations, actions)
        targets = targets.sum(-1)
        assertEquals(Q_values.size(), targets.size())
        loss = F.mse_loss(Q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters, self.grad_norm_clip)
        self.optimizer.step()

    def global_value(self, local_Q_values, max_Q_values, observations, actions):
        batch_size = int(observations.size(0)/self.nr_agents)
        observations = observations.view(batch_size, self.nr_agents, -1)
        V_i_local = max_Q_values
        A_i_local = local_Q_values - max_Q_values
        return self.mixing_network(V_i_local, A_i_local, observations, actions).squeeze()
    
def make(params):
    critic_name = params[CRITIC_NAME]
    if critic_name == CRITIC_VDN:
        return VDNCritic(params)
    if critic_name == CRITIC_QMIX:
        return QMIXCritic(params)
    if critic_name == CRITIC_CENTRAL:
        return CentralCritic(params)
    if critic_name == CRITIC_QPLEX:
        return QPLEXCritic(params)
    raise ValueError(f"Unknown critic: '{critic_name}'")