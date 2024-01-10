from cactus.controller.controller import Controller
from cactus.modules.primal_module import PRIMALModule
from cactus.constants import *
from torch.distributions import Categorical
from os.path import join

class PRIMALController(Controller):
    
    def __init__(self, params) -> None:
        super(PRIMALController, self).__init__(params)
        self.agent_network = PRIMALModule(params)
        self.hidden_states = self.agent_network.init_hidden(1)
        self.agent_parameters = self.agent_network.parameters()
        self.optimizer = torch.optim.Adam(self.agent_parameters, lr=self.learning_rate)

    def joint_policy(self, joint_observation):
        joint_observation = joint_observation.view(1, self.nr_agents, -1)
        self.hidden_states = self.hidden_states.view(1, self.nr_agents, -1)
        action_probs, _, self.hidden_states = self.agent_network(joint_observation, self.hidden_states)
        m = Categorical(action_probs.detach())
        joint_action = m.sample()
        return joint_action.view(self.nr_agents)
    
    def save_model_weights(self, path):
        actor_path = join(path, ACTOR_NET_FILENAME)
        torch.save(self.agent_network.state_dict(), actor_path)

    def load_model_weights(self, path):
        actor_path = join(path, ACTOR_NET_FILENAME)
        self.agent_network.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.agent_network.eval()
    
    def policy_loss(self, advantage, probs, action, old_probs=None):
        m1 = Categorical(probs)
        return -m1.log_prob(action)*advantage
    
    def reset_hidden_state(self):
        self.hidden_states = self.agent_network.init_hidden(1)

    def train(self):
        batch_size = self.episodes_per_epoch
        obs, actions, returns, dones, _ = self.memory.get_training_data()
        max_length = obs.size(0)
        returns = returns.sum(-1).unsqueeze(2).expand(max_length, batch_size, self.nr_agents)
        normalized_returns = (returns - returns.mean())/(returns.std() + EPSILON)
        probs = []
        values = []
        h = self.agent_network.init_hidden(self.episodes_per_epoch)
        for o in obs:
            p, v, h = self.agent_network(o.view(-1, self.nr_agents, self.agent_network.input_shape), h)
            h = h.view(batch_size, self.nr_agents, -1)
            probs.append(p)
            values.append(v.view(-1))
        probs = self.stack(probs).view(-1, self.nr_actions)
        values = self.stack(values)
        advantages = (normalized_returns.view(-1) - values.view(-1)).detach()
        mask = (1 - dones.to(INT_TYPE)).view(-1)
        masked_total = mask.sum()
        mask_all = (1 - dones.all(-1).to(INT_TYPE)).view(-1)
        masked_total_all = mask_all.sum()
        if self.vdn_mode:
            predicted_values = values.view(-1, self.nr_agents).sum(-1)
            observed_values = normalized_returns.view(-1, self.nr_agents).mean(-1)
            masked_value_error = (predicted_values - observed_values)*mask_all
            value_loss = (masked_value_error*masked_value_error).sum()/masked_total_all
        else:
            masked_value_error = (values.view(-1) - normalized_returns.view(-1))*mask
            value_loss = (masked_value_error*masked_value_error).sum()/masked_total
        masked_policy_loss = self.policy_loss(advantages, probs, actions.view(-1))*mask
        policy_loss = masked_policy_loss.sum()/masked_total
        loss = (policy_loss + value_loss)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent_parameters, self.grad_norm_clip)
        self.optimizer.step()

