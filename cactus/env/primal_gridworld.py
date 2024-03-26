from cactus.env.mapf_gridworld import MAPFGridWorld
from cactus.constants import *

"""
 Represents a navigation task in a 2D grid world environment with collision detection.
"""
class PRIMALGridWorld(MAPFGridWorld):

    def __init__(self, params) -> None:
        obstacles = params[ENV_PRIMAL_MAP][0].copy()
        params[ENV_NR_AGENTS] = numpy.max(obstacles)
        obstacles[obstacles >= 0] = 0
        params[ENV_OBSTACLES] = -obstacles
        params[ENV_TIME_LIMIT] = 256
        params[ENV_OBSERVATION_SIZE] = 7
        params[ENV_GAMMA] = 1
        super(PRIMALGridWorld, self).__init__(params)
        self.primal_map = params[ENV_PRIMAL_MAP].copy()

    def reset(self):
        super(PRIMALGridWorld, self).reset()
        self.occupied_goal_positions[:] = -1
        self.current_position_map[:] = -1.0
        self.time_step = 0
        self.discounted_returns[:] = 0
        self.undiscounted_returns[:] = 0
        self.joint_observation_buffer[:] = 0
        self.current_positions[:] = 0
        self.goal_positions[:] = 0
        for x in range(self.rows):
            for y in range(self.columns):
                start_value = self.primal_map[0][x][y]
                if start_value > 0:
                    self.current_positions[start_value-1] = self.int_tensor([x,y])
                goal_value = self.primal_map[1][x][y]
                if goal_value > 0:
                    goal_position = (x,y)
                    self.goal_positions[goal_value-1] = self.int_tensor(goal_position)
                    self.occupied_goal_positions[x][y] = goal_value-1
        return self.joint_observation()

