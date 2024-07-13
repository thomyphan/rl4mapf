from cactus.utils import assertContains, get_param_or_default
from cactus.tensorable import Tensorable
from cactus.constants import *

class Memory(Tensorable):

    def __init__(self, params) -> None:
        assertContains(params, ENV_NR_AGENTS)
        assertContains(params, ENV_NR_ACTIONS)
        assertContains(params, ENV_OBSERVATION_DIM)
        assertContains(params, TORCH_DEVICE)
        assertContains(params, ENV_TIME_LIMIT)
        assertContains(params, ENV_GAMMA)
        self.device = params[TORCH_DEVICE]
        super(Memory, self).__init__(self.device)
        self.observation_dim = params[ENV_OBSERVATION_DIM]
        self.observation_size = numpy.prod(self.observation_dim)
        self.nr_agents = params[ENV_NR_AGENTS]
        self.nr_actions = params[ENV_NR_ACTIONS]
        self.time_limit = params[ENV_TIME_LIMIT]
        self.gamma = params[ENV_GAMMA]
        self.episode_buffer = []
        self.joint_observations = []
        self.joint_actions = []
        self.joint_rewards = []
        self.joint_returns = []
        self.joint_dones = []
        self.max_episode_length = 0
        self.episode_count = 0
        self.time_step = 0

    def save(self, joint_observation, joint_action, joint_reward, joint_terminated, joint_truncated, done):
        self.episode_buffer.append((joint_observation, joint_action, joint_reward, joint_terminated, joint_truncated))
        self.time_step += 1
        if done:
            self.time_step = 0
            self.episode_count += 1
            episode_length = len(self.episode_buffer)
            assert episode_length <= self.time_limit, f"{episode_length} should not be greater than {self.time_limit}"
            if episode_length > self.max_episode_length:
                self.max_episode_length = episode_length
            returns = self.float_zeros(self.nr_agents)
            joint_observation_buffer = []
            joint_action_buffer = []
            joint_reward_buffer = []
            joint_return_buffer = []
            joint_done_buffer = []
            for transition in reversed(self.episode_buffer):
                obs, actions, rewards, dones, _ = transition
                returns = rewards + self.gamma*returns
                joint_action_buffer.append(actions)
                joint_observation_buffer.append(obs)
                joint_reward_buffer.append(rewards)
                joint_done_buffer.append(dones)
                joint_return_buffer.append(returns.detach().clone())
            joint_observation_buffer.reverse()
            joint_action_buffer.reverse()
            joint_reward_buffer.reverse()
            joint_return_buffer.reverse()
            joint_done_buffer.reverse()
            self.joint_observations.append(self.stack(joint_observation_buffer))
            self.joint_actions.append(self.stack(joint_action_buffer))
            self.joint_rewards.append(self.stack(joint_reward_buffer))
            self.joint_dones.append(self.stack(joint_done_buffer))
            self.joint_returns.append(self.stack(joint_return_buffer))
            self.episode_buffer.clear()

    def get_training_data(self, truncated=False):
        if truncated:
            return self.cat(self.joint_observations).view(-1, self.nr_agents, self.observation_size),\
                self.cat(self.joint_actions).view(-1, self.nr_agents),\
                self.cat(self.joint_returns).view(-1, self.nr_agents),\
                self.cat(self.joint_dones).view(-1, self.nr_agents),\
                self.cat(self.joint_rewards).view(-1, self.nr_agents)
        else:
            observation_tensor = self.float_zeros(\
                [self.max_episode_length, self.episode_count, self.nr_agents] + self.observation_dim)
            action_tensor = self.int_zeros([self.max_episode_length, self.episode_count, self.nr_agents])
            reward_tensor = self.float_zeros([self.max_episode_length, self.episode_count, self.nr_agents])
            return_tensor = self.float_zeros([self.max_episode_length, self.episode_count, self.nr_agents])
            done_tensor = self.bool_ones([self.max_episode_length, self.episode_count, self.nr_agents])
            index = 0
            for obs, action, reward, done, returns in\
                zip(self.joint_observations, self.joint_actions, self.joint_rewards, self.joint_dones, self.joint_returns):
                episode_length = obs.size(0)
                observation_tensor[:episode_length, index, :, :, :, :] = obs
                action_tensor[:episode_length, index, :] = action
                reward_tensor[:episode_length, index, :] = reward
                return_tensor[:episode_length, index, :] = returns
                done_tensor[:episode_length, index, :] = done
                index += 1
            return observation_tensor, action_tensor, return_tensor, done_tensor, reward_tensor

    def clear(self):
        self.joint_observations.clear()
        self.joint_actions.clear()
        self.joint_rewards.clear()
        self.joint_returns.clear()
        self.joint_dones.clear()
        self.max_episode_length = 0
        self.episode_count = 0

class Controller(Tensorable):

    def __init__(self, params) -> None:
        assertContains(params, ENV_NR_AGENTS)
        assertContains(params, ENV_NR_ACTIONS)
        assertContains(params, ENV_OBSERVATION_DIM)
        assertContains(params, TORCH_DEVICE)
        assertContains(params, EPISODES_PER_EPOCH)
        self.device = params[TORCH_DEVICE]
        super(Controller, self).__init__(self.device)
        self.observation_dim = params[ENV_OBSERVATION_DIM]
        self.nr_agents = params[ENV_NR_AGENTS]
        self.nr_actions = params[ENV_NR_ACTIONS]
        self.episodes_per_epoch = params[EPISODES_PER_EPOCH]
        self.memory = Memory(params)
        self.episode_count = 0
        self.grad_norm_clip = get_param_or_default(params, GRAD_NORM_CLIP, 1)
        self.learning_rate = get_param_or_default(params, LEARNING_RATE, 0.001)
        self.vdn_mode = get_param_or_default(params, VDN_MODE, False)
        self.reward_sharing = get_param_or_default(params, REWARD_SHARING, True)
        self.agent_ids = [i for i in range(self.nr_agents)]
        self.grid_operations = self.int_zeros((NR_GRID_ACTIONS, ENV_2D))
        self.grid_operations[WAIT]  = self.as_int_tensor([ 0,  0])
        self.grid_operations[NORTH] = self.as_int_tensor([ 0,  1])
        self.grid_operations[SOUTH] = self.as_int_tensor([ 0, -1])
        self.grid_operations[WEST]  = self.as_int_tensor([-1,  0])
        self.grid_operations[EAST]  = self.as_int_tensor([ 1,  0])
    
    def get_parameter_count(self):
        return 0

    def save_model_weights(self, path):
        pass

    def load_model_weights(self, path):
        pass

    def joint_policy(self, joint_observation):
        return torch.randint(0, self.nr_actions, (self.nr_agents,))

    def train(self):
        pass

    def reset_hidden_state(self):
        pass

    def update(self, joint_observation, joint_action, joint_reward, joint_terminated, joint_truncated, done, info):
        self.memory.save(joint_observation, joint_action, joint_reward, joint_terminated, joint_truncated, done)
        if done:
            self.episode_count += 1
            if self.episode_count > 0 and self.episode_count%self.episodes_per_epoch == 0:
                self.train()
                self.memory.clear()
                self.episode_count = 0
            self.reset_hidden_state()