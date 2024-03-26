from cactus.env.gridworld import GridWorld
from cactus.constants import *

"""
 Represents a navigation task in a 2D grid world environment with collision detection.
"""
class CollisionGridWorld(GridWorld):

    def __init__(self, params) -> None:
        super(CollisionGridWorld, self).__init__(params)
        self.agent_ids = self.as_int_tensor([i for i in range(self.nr_agents)])
        self.vertex_collision_buffer = self.bool_ones(self.nr_agents)
        self.edge_collision_buffer = self.bool_ones(self.nr_agents)

    def move_condition(self, new_positions):
        self.current_position_map[:] = -1.0
        condition, _ = super(CollisionGridWorld, self).move_condition(new_positions)
        condition = condition.all(1)
        self.edge_collision_buffer.fill_(False)
        self.vertex_collision_buffer.fill_(False)
        x0 = self.current_positions[:,0]
        y0 = self.current_positions[:,1]
        self.current_position_map[x0,y0] = self.agent_ids
        x1 = torch.where(condition, new_positions[:,0], self.current_positions[:,0])
        y1 = torch.where(condition, new_positions[:,1], self.current_positions[:,1])
        self.next_position_map[x1,y1] = self.agent_ids
        self.vertex_collision_buffer[:] = (self.next_position_map[x1,y1] != self.agent_ids)
        other_origins = -self.int_ones(self.nr_agents)
        other_origins = self.current_position_map[x1,y1]
        occupied = other_origins >= 0
        filter_condition = condition[other_origins.clamp(0, self.nr_agents)]
        other_origins = torch.where(torch.logical_and(filter_condition, occupied), other_origins, -1)
        occupied = other_origins >= 0
        not_same = other_origins != self.agent_ids
        edge_condition = torch.logical_and(occupied, not_same)
        indices = other_origins[edge_condition]
        x = new_positions[indices,0]
        y = new_positions[indices,1]
        if edge_condition.any():
            self.edge_collision_buffer[edge_condition] = self.current_position_map[x,y] == self.agent_ids[edge_condition]
        no_collisions = torch.logical_not(torch.logical_or(self.vertex_collision_buffer, self.edge_collision_buffer))
        condition = torch.logical_and(condition, no_collisions)
        return condition.unsqueeze(1).expand(self.nr_agents, ENV_2D), (self.vertex_collision_buffer, self.edge_collision_buffer)

    def reset(self):
        self.current_position_map[:] = -1.0
        return super(CollisionGridWorld, self).reset()