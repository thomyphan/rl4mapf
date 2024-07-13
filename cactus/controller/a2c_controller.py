from cactus.controller.controller import Controller
from cactus.modules.ffn_module import FFNModule
from cactus.utils import get_param_or_default, assertEquals
from cactus.constants import *
from torch.distributions import Categorical
from os.path import join
import torch.nn.functional as F
import cactus.controller.critic as critic

class A2CController(Controller):
    
    def __init__(self, params) -> None:
        super(A2CController, self).__init__(params)
        params[OUTPUT_DIM] = self.nr_actions
        self.policy_network = FFNModule(params)
        self.clip_ratio = get_param_or_default(params, CLIP_RATIO, 0.1)
        self.update_iterations = get_param_or_default(params, UPDATE_ITERATIONS, 1)
        self.policy_parameters = self.policy_network.parameters()
        self.policy_optimizer = torch.optim.Adam(self.policy_parameters, lr=self.learning_rate)
        self.has_critic = CRITIC_NAME in params
        if self.has_critic:
            self.critic_network = critic.make(params)
        else:
            params[OUTPUT_DIM] = 1
            self.critic_network = FFNModule(params)
            self.critic_parameters = self.critic_network.parameters()
            self.critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=self.learning_rate)

    def calculate_action_masks(self, joint_observation):
        channels, o_size, o_size = self.observation_dim[0], self.observation_dim[1], self.observation_dim[2]
        batch_size = joint_observation.size(0)
        joint_observation = joint_observation.view(batch_size, self.nr_agents, channels, o_size, o_size)
        half_size = int(self.observation_dim[-1]/2)
        invalid_actions = self.bool_zeros([batch_size, self.nr_agents, self.nr_actions])
        for i, delta in enumerate(self.grid_operations[GRID_ACTIONS]):
            dx,dy = delta[0], delta[1]
            invalid_actions[:,self.agent_ids,i] = joint_observation[:,self.agent_ids,2,half_size+dx,half_size+dy].to(torch.bool)
        action_mask = self.float_zeros_like(invalid_actions)
        action_mask[invalid_actions] = float('-inf')
        return action_mask.view(-1, self.nr_actions)

    def get_parameter_count(self):
        nr_actor_params = sum(p.numel() for p in self.policy_network.parameters() if p.requires_grad)
        if self.has_critic:
            self.critic_network.get_parameter_count()
        else:
            nr_critic_params = sum(p.numel() for p in self.critic_network.parameters() if p.requires_grad)
        return nr_actor_params + nr_critic_params

    def save_model_weights(self, path):
        actor_path = join(path, ACTOR_NET_FILENAME)
        torch.save(self.policy_network.state_dict(), actor_path)
        if not self.has_critic:
            critic_path = join(path, CRITIC_NET_FILENAME)
            torch.save(self.critic_network.state_dict(), critic_path)
        else:
            self.critic_network.save_model_weights(path)

    def load_model_weights(self, path):
        actor_path = join(path, ACTOR_NET_FILENAME)
        self.policy_network.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.policy_network.eval()
        if not self.has_critic:
            critic_path = join(path, CRITIC_NET_FILENAME)
            self.critic_network.load_state_dict(torch.load(critic_path, map_location=self.device))
            self.critic_network.eval()
        else:
            self.critic_network.load_model_weights(path)

    def joint_policy(self, joint_observation):
        joint_observation = joint_observation.view(1, self.nr_agents, -1)
        action_mask = self.calculate_action_masks(joint_observation)
        action_logits = self.policy_network(joint_observation)
        assertEquals(action_mask.size(), action_logits.size())
        probs = F.softmax(action_logits+action_mask, dim=-1).detach()
        m = Categorical(probs)
        joint_action = m.sample()
        return joint_action.view(self.nr_agents)
    
    def policy_loss(self, advantage, probs, action, old_probs=None):
        m1 = Categorical(probs)
        return -m1.log_prob(action)*advantage
    
    def update_critic(self, observation, actions, targets):
        if self.has_critic:
            self.critic_network.train(observation, actions, targets)
        else:
            values = self.critic_network(observation).view(-1)
            observed_values = targets.view(-1)
            error = (values - observed_values)
            value_loss = (error*error).mean()
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.grad_norm_clip)
            self.critic_optimizer.step()

    def train(self):
        obs, actions, returns, _, _ = self.memory.get_training_data(truncated=True)
        action_mask = self.calculate_action_masks(obs)
        obs = obs.view(-1, self.policy_network.input_shape)
        actions = actions.view(-1)
        returns = returns.view(-1)
        returns = (returns - returns.mean())/(returns.std() + EPSILON)
        action_logits = self.policy_network(obs)
        probs = F.softmax(action_logits+action_mask, dim=-1)
        old_probs = probs.detach()
        for i in range(self.update_iterations):
            self.update_critic(obs, actions, returns)
            if self.has_critic:
                baseline = self.critic_network.counterfactual_baseline(obs, actions, probs).view(-1)
            else:
                baseline = self.critic_network(obs).view(-1)
            advantages = (returns.view(-1) - baseline).detach()
            policy_loss = self.policy_loss(advantages, probs, actions.view(-1), old_probs).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_parameters, self.grad_norm_clip)
            self.policy_optimizer.step()
            if i+1 < self.update_iterations:
                action_logits = self.policy_network(obs)
                probs = F.softmax(action_logits+action_mask, dim=-1)

