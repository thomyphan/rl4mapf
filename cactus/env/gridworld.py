from cactus.env.environment import Environment
from cactus.utils import assertContains, assertEquals, get_param_or_default
from cactus.constants import *
from cactus.rendering.gridworld_viewer import render
import torch
import random
import heapq

"""
 Represents a 2D grid world
"""
class GridWorld(Environment):

    def __init__(self, params) -> None:
        params[ENV_NR_ACTIONS] = NR_GRID_ACTIONS
        assertContains(params, ENV_OBSTACLES)
        super(GridWorld, self).__init__(params)
        self.makespan_mode = get_param_or_default(params, ENV_MAKESPAN_MODE, False)
        self.obstacle_map = self.as_bool_tensor(params[ENV_OBSTACLES])
        self.rows = self.obstacle_map.size(0)
        self.columns = self.obstacle_map.size(1)
        self.grid_operations = self.int_zeros((NR_GRID_ACTIONS, ENV_2D))
        self.grid_operations[WAIT]  = self.as_int_tensor([ 0,  0])
        self.grid_operations[NORTH] = self.as_int_tensor([ 0,  1])
        self.grid_operations[SOUTH] = self.as_int_tensor([ 0, -1])
        self.grid_operations[WEST]  = self.as_int_tensor([-1,  0])
        self.grid_operations[EAST]  = self.as_int_tensor([ 1,  0])
        self.delta_to_actions = {
            ( 0, 0) : WAIT,
            ( 0, 1) : NORTH,
            ( 0,-1) : SOUTH,
            (-1, 0) : WEST,
            ( 1, 0) : EAST
        }
        self.occupiable_locations = self.get_occupiable_locations()
        self.shortest_distance_maps = {}
        self.current_positions = self.int_zeros([self.nr_agents, ENV_2D])
        self.goal_positions = self.int_zeros([self.nr_agents, ENV_2D])
        self.init_goal_positions = get_param_or_default(params, ENV_GOAL_POSITIONS, None)
        self.init_start_positions = get_param_or_default(params, ENV_START_POSITIONS, None)
        self.collision_weight = get_param_or_default(params, ENV_COLLISION_WEIGHT, 0)
        self.time_penalty = get_param_or_default(params, ENV_TIME_PENALTY, -1.0)
        self.init_goal_radius = get_param_or_default(params, ENV_INIT_GOAL_RADIUS, None)
        self.completion_reward = get_param_or_default(params, ENV_COMPLETION_REWARD, 1.0)
        self.use_primal_reward = get_param_or_default(params, ENV_USE_PRIMAL_REWARD, False)
        if self.use_primal_reward:
            self.collision_weight = 2
            self.time_penalty = -0.3
            self.completion_reward = 20.0
        self.occupied_goal_positions = -self.int_ones([self.rows, self.columns])
        self.shortest_distance_map = -self.int_ones([self.nr_agents, self.rows, self.columns])
        self.current_position_map = -self.int_ones_like(self.obstacle_map)
        self.next_position_map = -self.int_ones_like(self.obstacle_map)
        self.viewer = None

    def has_init_configuration(self):
        return self.init_goal_positions is not None and self.init_start_positions is not None

    def render(self):
        self.viewer = render(self, self.viewer)

    def get_occupiable_locations(self):
        return [(r,c) for r in range(self.rows) for c in range(self.columns) if not self.obstacle_map[r][c]]
    
    def get_neighbor_positions(self, position, delta):
        assert delta > 0
        x, y = position[0], position[1]
        neighbors = []
        for dx in range(-delta, delta+1):
            for dy in range(-delta, delta+1):
                new_pos = (x+dx, y+dy)
                x1, y1 = new_pos
                no_overlap = (dx, dy) != (0,0)
                in_bounds = x1 >= 0 and y1 >= 0 and x1 < self.rows and y1 < self.columns 
                if no_overlap and in_bounds and not self.obstacle_map[x1][y1]:
                    neighbors.append(new_pos)
        return neighbors

    def step(self, joint_action):
        self.time_step += 1
        if self.time_step >= self.time_limit or self.is_done().all():
            assert self.is_done().all()
            terminated = self.is_terminated()
            return self.joint_observation(), self.float_zeros(self.nr_agents), terminated, self.is_truncated(), {
                ENV_VERTEX_COLLISIONS: self.bool_zeros(self.nr_agents),
                ENV_EDGE_COLLISIONS: self.bool_zeros(self.nr_agents),
                ENV_COMPLETION_RATE: terminated.to(FLOAT_TYPE).sum()/self.nr_agents
            }
        is_done_before = self.is_terminated()
        not_done = torch.logical_not(is_done_before)
        joint_action = self.grid_operations[joint_action]
        new_positions = self.current_positions + joint_action
        self.current_positions, collisions = self.move_to(new_positions)
        self.current_position_map[:] = -1.0
        x = self.current_positions[:,0]
        y = self.current_positions[:,1]
        self.current_position_map[x,y] = self.agent_ids
        is_done_now = self.is_terminated()
        if self.makespan_mode:
            reward = self.float_ones(self.nr_agents)
            if not self.is_truncated().all():
                reward *= -1
        else:
            reward = torch.where(torch.logical_and(is_done_now, not_done), 1.0, self.time_penalty).to(FLOAT_TYPE)
            was_done = torch.logical_and(is_done_now, is_done_before)
            reward = torch.where(torch.logical_and(is_done_now, is_done_before), self.float_zeros_like(reward), reward)
            if self.use_primal_reward:
                was_not_done = torch.logical_not(was_done)
                reward = torch.where(torch.logical_and(was_not_done, (joint_action.sum(-1)==0).all()), -0.5, reward)
        vertex_collisions, edge_collisions = collisions
        if self.collision_weight is not None:
            vertex_collisions = vertex_collisions.to(INT_TYPE)
            edge_collisions = edge_collisions.to(INT_TYPE)/2
            reward -= self.collision_weight*(vertex_collisions + edge_collisions).to(FLOAT_TYPE)
        terminated = self.is_terminated()
        if terminated.all():
            reward += self.completion_reward
        self.undiscounted_returns += reward
        self.discounted_returns += (self.gamma**(self.time_step-1))*reward
        return self.joint_observation(), reward, terminated, self.is_truncated(),\
            {
                ENV_VERTEX_COLLISIONS: vertex_collisions,
                ENV_EDGE_COLLISIONS: edge_collisions,
                ENV_COMPLETION_RATE: terminated.to(FLOAT_TYPE).sum()/self.nr_agents
            }

    def is_terminated(self):
        x_equal = self.current_positions[:,0] == self.goal_positions[:,0]
        y_equal = self.current_positions[:,1] == self.goal_positions[:,1]
        return torch.logical_and(x_equal, y_equal)
    
    def move_condition(self, new_positions):
        in_bounds = self.position_in_bounds(new_positions)
        return in_bounds.unsqueeze(1).expand(self.nr_agents, ENV_2D), (self.bool_zeros(self.nr_agents), self.bool_zeros(self.nr_agents))

    def move_to(self, new_positions):
        new_positions_changed = True
        vertex_collisions = self.bool_zeros(self.nr_agents)
        edge_collisions = self.bool_zeros(self.nr_agents)
        while new_positions_changed:
            condition, new_collisions = self.move_condition(new_positions)
            new_positions_1 = torch.where(condition, new_positions, self.current_positions)
            vertex_collisions = torch.logical_or(vertex_collisions, new_collisions[0])
            edge_collisions = torch.logical_or(edge_collisions, new_collisions[1])
            new_positions_changed = (new_positions_1 != new_positions).any()
            new_positions = new_positions_1
        return torch.where(condition, new_positions, self.current_positions), (vertex_collisions, edge_collisions)

    def position_in_bounds(self, pos):
        x = pos[:,0]
        y = pos[:,1]
        return self.xy_position_in_bounds(x, y)

    def set_init_goal_radius(self, radius):
        self.init_goal_radius = radius

    def increment_init_goal_radius(self):
        self.init_goal_radius += 1
    
    def decrement_init_goal_radius(self):
        self.init_goal_radius -= 1
    
    def xy_position_in_bounds(self, x, y):
        nonnegative = torch.logical_and(x >= 0, y >= 0)
        in_bounds = torch.logical_and(x < self.rows, y < self.columns)
        in_bounds = torch.logical_and(in_bounds, nonnegative)
        x_clamped = x.clamp(0, self.rows-1)
        y_clamped = y.clamp(0, self.columns-1)
        no_obstacle = torch.logical_not(self.obstacle_map[x_clamped, y_clamped])
        return torch.logical_and(in_bounds, no_obstacle)
    
    def get_neighbor_positions(self, position, delta):
        assert delta > 0
        x, y = position[0], position[1]
        neighbors = []
        for dx in range(-delta, delta+1):
            for dy in range(-delta, delta+1):
                new_pos = (x+dx, y+dy)
                x1, y1 = new_pos
                no_overlap = (dx, dy) != (0,0)
                in_bounds = x1 >= 0 and y1 >= 0 and x1 < self.rows and y1 < self.columns 
                if in_bounds:
                    no_obstacle = not self.obstacle_map[x1][y1]
                    not_occupied = self.occupied_goal_positions[x1][y1] < 0
                else:
                    no_obstacle = False
                    not_occupied = False
                if no_overlap and in_bounds and no_obstacle and not_occupied:
                    neighbors.append(new_pos)
        return neighbors
    
    def set_start_positions(self, positions):
        for a in range(self.nr_agents):
            self.current_positions[a,0] = positions[a][0]
            self.current_positions[a,1] = positions[a][1]

    def set_goal_positions(self, positions):
        self.occupied_goal_positions[:] = -1
        for a in range(self.nr_agents):
            self.goal_positions[a,0] = positions[a][0]
            self.goal_positions[a,1] = positions[a][1]
            self.occupied_goal_positions[positions[a][0]][positions[a][1]] = a
    
    def reset(self):
        super(GridWorld, self).reset()
        self.occupied_goal_positions[:] = -1
        random.shuffle(self.occupiable_locations)
        nr_samples = 2*self.nr_agents
        sampled_locations = random.sample(self.occupiable_locations, k=nr_samples)
        for a in range(self.nr_agents):
            index = a*2
            x, y = sampled_locations[index][0], sampled_locations[index][1]
            self.current_positions[a,0] = x
            self.current_positions[a,1] = y
            if self.init_goal_radius is not None and self.init_goal_radius < max(self.rows, self.columns):
                goal_candidates = self.get_neighbor_positions((x, y), self.init_goal_radius)
                sampled_location = random.choice(goal_candidates)
                self.goal_positions[a,0] = sampled_location[0]
                self.goal_positions[a,1] = sampled_location[1]
                self.occupied_goal_positions[sampled_location[0]][sampled_location[1]] = a
            else:
                self.goal_positions[a,0] = sampled_locations[index+1][0]
                self.goal_positions[a,1] = sampled_locations[index+1][1]
                self.occupied_goal_positions[self.goal_positions[a,0]][self.goal_positions[a,1]] = a
        assertEquals(self.nr_agents, len(self.occupied_goal_positions[self.occupied_goal_positions >= 0].view(-1)))
        return self.joint_observation()
    
    def get_adjacent_neighbors(self, pos):
        x, y = pos
        neighbors = []
        if x > 0 and not self.obstacle_map[x-1][y]:
            neighbors.append((x-1, y))
        if x < self.rows-1 and not self.obstacle_map[x+1][y]:
            neighbors.append((x+1, y))
        if y > 0 and not self.obstacle_map[x][y-1]:
            neighbors.append((x, y-1))
        if y < self.columns-1 and not self.obstacle_map[x][y+1]:
            neighbors.append((x, y+1))
        return neighbors
    
    def compute_shortest_distances(self):
        for i in range(self.nr_agents):
            goal_position = (self.goal_positions[i,0].item(), self.goal_positions[i,1].item())
            self.shortest_distance_map[i,:,:] = self.shortest_distance_maps[goal_position]

    def compute_shortest_distances_for(self, map_tensor, goal_position):
        x, y = goal_position
        pos = (x, y)
        queue = [(0, pos)]
        while len(queue) > 0:
            current_distance, current_vertex = heapq.heappop(queue)
            x0, y0 = current_vertex
            stored_distance = map_tensor[x0][y0]
            if stored_distance < 0 or current_distance <= stored_distance:
                x0, y0 = current_vertex
                map_tensor[x0][y0] = current_distance
                for neighbor in self.get_adjacent_neighbors(current_vertex):
                    distance = current_distance + 1
                    x1, y1 = neighbor
                    stored_distance = map_tensor[x1][y1]
                    if stored_distance < 0 or distance < stored_distance:
                        x1, y1 = neighbor
                        map_tensor[x1][y1] = distance
                        heapq.heappush(queue, (distance, neighbor))
        return map_tensor

    def print(self):
        map_tensor = self.int_zeros_like(self.obstacle_map) - self.obstacle_map.to(INT_TYPE)
        x = self.current_positions[:,0]
        y = self.current_positions[:,1]
        map_tensor[x,y] = 1
        x = self.goal_positions[:,0]
        y = self.goal_positions[:,1]
        map_tensor[x,y] = 2
        for x in range(self.rows):
            line = ""
            for y in range(self.columns):
                if map_tensor[x][y] < 0:
                    line += "# "
                elif map_tensor[x][y] == 1:
                    line += "O "
                elif map_tensor[x][y] == 2:
                    line += "X "
                else:
                    line += ". "
            print(line)
