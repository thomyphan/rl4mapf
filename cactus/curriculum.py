from cactus.utils import assertContains, get_param_or_default
from cactus.constants import *

class Curriculum:

    def __init__(self, envs) -> None:
        self.envs = envs

    def update_curriculum(self, objective_value, variance):
        pass

    def get_improvement_threshold(self):
        return 0.0

    def init_goal_radius(self):
        return -1.0

class RandomCurriculum(Curriculum):

    def __init__(self, env, params) -> None:
        super(RandomCurriculum, self).__init__(env)

    def update_curriculum(self, objective_value, variance):
        for env in self.envs:
            env.set_init_goal_radius(None)

class CACTUSCurriculum(Curriculum):
    def __init__(self, envs, params) -> None:
        super(CACTUSCurriculum, self).__init__(envs)
        assertContains(params, RADIUS_UPDATE_INTERVAL)
        self.deviation_factor = get_param_or_default(params, DEVIATION_FACTOR, 2) # 97% confidence as default
        self.radius = 2
        for env in self.envs:
            env.set_init_goal_radius(self.radius)
        self.objective_values = []
        self.epoch_count = 0
        self.value_count = 0
        self.total_sum = 0.0
        self.total_sum_squared = 0.0
        self.sliding_window_size = get_param_or_default(params, SLIDING_WINDOW_SIZE, 50)
        self.improvement_threshold = get_param_or_default(params, IMPROVEMENT_THRESHOLD, 0.75)
    
    def get_improvement_threshold(self):
        return self.improvement_threshold

    def init_goal_radius(self):
        return self.radius

    def update_curriculum(self, objective_value, variance):
        # Keep track of progress
        self.objective_values.append(objective_value)
        self.value_count += 1
        self.total_sum += objective_value
        self.total_sum_squared += (objective_value*objective_value)
        if self.value_count >= self.sliding_window_size:
            mean = self.total_sum/self.sliding_window_size
            stddev = numpy.sqrt(self.total_sum_squared/self.sliding_window_size - mean*mean)
            oldest_value = self.objective_values.pop(0)
            self.total_sum -= oldest_value
            self.total_sum_squared -= (oldest_value*oldest_value)
            if mean - self.deviation_factor*stddev >= self.improvement_threshold:
                self.adjust_threshold(mean, stddev)
                self.radius += 1
                for env in self.envs:
                    env.set_init_goal_radius(self.radius)
        self.epoch_count += 1

    def adjust_threshold(self, mean, stddev):
        pass

def make(params):
    curriculum_name = params[CURRICULUM_NAME]
    if curriculum_name == RANDOM_CURRICULUM:
        return RandomCurriculum
    if curriculum_name == CACTUS_CURRICULUM:
        return CACTUSCurriculum
    raise ValueError(f"Unknown curriculum: '{curriculum_name}'")