from cactus.env.collision_gridworld import CollisionGridWorld
from cactus.utils import assertContains
from cactus.constants import *

"""
 Represents a navigation task in a 2D grid world environment with collision detection.
"""
class MAPFGridWorld(CollisionGridWorld):

    def __init__(self, params) -> None:
        assertContains(params, ENV_OBSERVATION_SIZE)
        self.nr_channels = 5
        self.observation_size = params["observation_size"]
        params[ENV_OBSERVATION_DIM] = [self.nr_channels, self.observation_size, self.observation_size]
        super(MAPFGridWorld, self).__init__(params)
        self.obseveration_dx, self.obseveration_dy = self.get_delta_tensor(int(self.observation_size/2))
        self.zero_observation = self.float_zeros((self.nr_agents, self.nr_channels, self.observation_size, self.observation_size))
        self.one_observation = self.float_ones((self.nr_agents, self.observation_size, self.observation_size))
        self.current_position_map = -self.int_ones_like(self.obstacle_map)
        self.next_position_map = -self.int_ones_like(self.obstacle_map)
        self.vertex_collision_buffer = self.bool_ones(self.nr_agents)
        self.edge_collision_buffer = self.bool_ones(self.nr_agents)
        self.center_mask = self.float_zeros((self.nr_agents, self.observation_size, self.observation_size))
        half_size = int(self.observation_size/2)
        self.center_mask[:,half_size, half_size] = 1.0
        self.center_mask[:,half_size+1, half_size] = 1.0
        self.center_mask[:,half_size-1, half_size] = 1.0
        self.center_mask[:,half_size, half_size+1] = 1.0
        self.center_mask[:,half_size, half_size-1] = 1.0

    def joint_observation(self):
        obs = super(MAPFGridWorld, self).joint_observation()\
            .view(self.nr_agents, self.nr_channels, self.observation_size, self.observation_size)
        obs[:] = 0
        self.current_position_map[:] = -1
        done = self.is_done()
        done = done.unsqueeze(1)\
            .expand(-1, self.nr_channels*self.observation_size*self.observation_size)\
            .view(-1, self.nr_channels, self.observation_size, self.observation_size)
        half_size = int(self.observation_size/2)
        x0 = self.current_positions[:,0]
        y0 = self.current_positions[:,1]
        self.current_position_map[x0, y0] = self.agent_ids
        x1 = self.goal_positions[:,0]
        y1 = self.goal_positions[:,1]
        dx = x1 - x0
        dy = y1 - y0
        abs_dx = torch.abs(dx)
        abs_dy = torch.abs(dy)
        manhattan_distance = abs_dx + abs_dy
        euclidean_distance = torch.sqrt(dx*dx + dy*dy)
        max_distance = torch.maximum(abs_dx, abs_dy)
        goal_in_sight = max_distance <= half_size

        # Scan position relative to the goal
        x_direction = torch.sign(dx).to(dtype=INT_TYPE)+half_size
        y_direction = torch.sign(dy).to(dtype=INT_TYPE)+half_size
        obs[self.agent_ids,0, x_direction, half_size] = abs_dx/euclidean_distance
        obs[self.agent_ids,0, half_size, y_direction] = abs_dy/euclidean_distance
        obs[self.agent_ids,0, half_size, half_size] = manhattan_distance.to(dtype=FLOAT_TYPE)
        obs[goal_in_sight,1,dx[goal_in_sight]+half_size, dy[goal_in_sight]+half_size] = 1

        # Scan surrounding obstacles and boundaries
        dx = (x0.unsqueeze(1) + self.obseveration_dx).view(-1)
        dy = (y0.unsqueeze(1) + self.obseveration_dy).view(-1)
        in_bounds = self.xy_position_in_bounds(dx, dy).view(self.nr_agents, self.observation_size, self.observation_size)
        obs[self.agent_ids,2,:,:] = torch.where(in_bounds, 0.0, 1.0)

        # Scan surrounding agents and their manhattan distances to their goals
        x_clamped = dx.clamp(0, self.rows-1)
        y_clamped = dy.clamp(0, self.columns-1)

        zero_obs = self.zero_observation[:,0,:,:]
        neighbor_ids = self.current_position_map[x_clamped, y_clamped]
        is_agents_position = torch.logical_and(neighbor_ids >= 0, in_bounds.view(-1))
        neighbor_ids = neighbor_ids[is_agents_position]
        if neighbor_ids.any():
            neighbor_condition = is_agents_position.view(self.nr_agents, self.observation_size, self.observation_size)
            obs[self.agent_ids,3,:,:] = torch.where(neighbor_condition, 1.0, 0.0)
            flattened_view = obs[self.agent_ids,3].view(-1)
            flattened_view[neighbor_condition.view(-1)] = manhattan_distance[neighbor_ids].to(FLOAT_TYPE) + 1
            obs[self.agent_ids,3,:,:] = flattened_view.view(self.nr_agents, self.observation_size, self.observation_size)
            obs[self.agent_ids, 3, half_size, half_size] = 0.0
            obs[self.agent_ids,0,:,:] = torch.where(obs[self.agent_ids,3,:,:] > 0, zero_obs, obs[self.agent_ids,0,:,:])
            obs[self.agent_ids,2,:,:] = torch.where(obs[self.agent_ids,3,:,:] > 0, self.one_observation, obs[self.agent_ids,2,:,:])
        
        # Scan surrounding goals and their manhattan distances to their respective agents
        goal_ids = self.occupied_goal_positions[x_clamped, y_clamped]
        is_goal_position = torch.logical_and(goal_ids >= 0, in_bounds.view(-1))
        goal_ids = goal_ids[is_goal_position]
        if goal_ids.any():
            goal_condition = is_goal_position.view(self.nr_agents, self.observation_size, self.observation_size)
            obs[self.agent_ids,4,:,:] = torch.where(goal_condition, 1.0, 0.0)
            flattened_view = obs[self.agent_ids,4].view(-1)
            distances = manhattan_distance[goal_ids].to(FLOAT_TYPE) + 1
            flattened_view[goal_condition.view(-1)] = distances
            obs[self.agent_ids,4,:,:] = flattened_view.view(self.nr_agents, self.observation_size, self.observation_size)
            max_distance = distances.max()
            obs[self.agent_ids, 4, :, :] -= obs[self.agent_ids, 1, :, :]
            template = obs[self.agent_ids, 4, :, :]
            obs[self.agent_ids, 4, :, :] = torch.maximum(template, self.float_zeros_like(template))
        return obs
    
    def get_delta_tensor(self, delta):
        assert delta > 0
        x = []
        y = []
        for _ in range(self.nr_agents):
            for dx in range(-delta, delta+1):
                for dy in range(-delta, delta+1):
                    x.append(dx)
                    y.append(dy)
        return self.as_int_tensor(x).view(self.nr_agents, -1),\
               self.as_int_tensor(y).view(self.nr_agents, -1)