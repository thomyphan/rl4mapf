import torch
import numpy

INTEGER_MAX_VALUE = 2147483647

# Environment Constants
ENV_TIME_LIMIT = "time_limit"
ENV_NR_AGENTS = "nr_agents"
ENV_NR_ACTIONS = "nr_actions"
ENV_OBSERVATION_DIM = "observation_dim" # Array of dimension indicating the shape [C,K,K] of a grid observation
ENV_OBSERVATION_SIZE = "observation_size" # Indicates the length K of a KxK grid observation
ENV_GAMMA = "gamma"
ENV_OBSTACLES = "obstacle_map"
ENV_COLLISION_WEIGHT = "collision_weight"
ENV_MAKESPAN_MODE = "makespan_mode"
ENV_NR_INIT_THREAD = "nr_init_threads"
ENV_2D = 2
ENV_PRIMAL_MAP = "primal_map"
ENV_COMPLETION_REWARD = "completion_reward"
ENV_VERTEX_COLLISIONS = "vertex_collisions"
ENV_EDGE_COLLISIONS = "edge_collisions"
ENV_INIT_GOAL_RADIUS = "init_goal_radius"
ENV_COMPLETION_RATE = "completion_rate"
ENV_START_POSITIONS = "start_positions"
ENV_GOAL_POSITIONS = "goal_positions"
TEST_INIT_GOAL_RADIUS = "test_init_goal_radius"
MAP_NAME = "map_name"
INSTANCE_FOLDER = "instance_folder"
DEFAULT_FOLDER = "instances"
ENV_TIME_PENALTY = "time_penalty"
ENV_USE_PRIMAL_REWARD = "use_primal_reward"

# Gridworld Constants
WAIT  = 0
NORTH = 1
SOUTH = 2
WEST  = 3
EAST  = 4
GRID_ACTIONS = [WAIT, NORTH, SOUTH, WEST, EAST]
NR_GRID_ACTIONS = len(GRID_ACTIONS)

# Algorithm Constants
ALGORITHM_NAME = "algorithm_name"
DEFAULT_ALGORITHM = "Random"
ALGORITHM_RANDOM = DEFAULT_ALGORITHM
ALGORITHM_A2C_RECURRENT = "A2CRecurrent"
ALGORITHM_A2C = "A2C"
ALGORITHM_A2C_VDN = "A2C_VDN"
ALGORITHM_A2C_QMIX = "A2C_QMIX"
ALGORITHM_A2C_QPLEX = "A2C_QPLEX"
ALGORITHM_PPO = "PPO"
ALGORITHM_MAPPO = "MAPPO"
ALGORITHM_PRIMAL = "PRIMAL"
ALGORITHM_PPO_VDN = "PPO_VDN"
ALGORITHM_PPO_QMIX = "PPO_QMIX"
ALGORITHM_PPO_QPLEX = "PPO_QPLEX"
CRITIC_NAME = "critic_name"
CRITIC_VDN = "VDN"
CRITIC_QMIX = "QMIX"
CRITIC_QPLEX = "QPLEX"
CRITIC_CENTRAL = "CentralQ"

# Torch and Numpy Constants
TORCH_DEVICE = "device"
FLOAT_TYPE = torch.float32
INT_TYPE = torch.long
BOOL_TYPE = torch.bool
EPSILON = numpy.finfo(numpy.float32).eps

#Experiment Constants
ACTOR_NET_FILENAME = "actor_net.pth"
CRITIC_NET_FILENAME = "critic_net.pth"
MIXER_NET_FILENAME = "mixer_net.pth"
DISCOUNTED_RETURNS = "discounted_returns"
UNDISCOUNTED_RETURNS = "undiscounted_returns"
AUC_COMPLETION = "auc_completion"
AUC_SUCCESS = "auc_success"
COMPLETION_RATE = ENV_COMPLETION_RATE
VERTEX_COLLISIONS = ENV_VERTEX_COLLISIONS
EDGE_COLLISIONS = ENV_EDGE_COLLISIONS
NUMBER_OF_EPOCHS = "nr_epochs"
EPISODES_PER_EPOCH = "episodes_per_epoch"
EPOCH_LOG_INTERVAL = "epoch_log_interval"
TRAINING_TIME = "training_time"
ALGORITHM_NAME = "algorithm_name"
MAP_NAME = "map_name"
DIRECTORY = "directory"
TOTAL_TIME = "total_time"
TIME_PER_EPOCH = "time_per_epoch"
RENDER_MODE = "render_mode"
TERMINATED = "terminated"
DATA_PREFIX_PATTERN = "data_prefix_pattern"
STATS_LABEL = "stats_label"
PLOT_TITLE = "title"

# Algorithm Hyperparameters
HIDDEN_LAYER_DIM = "hidden_layer_dim"
NR_ITERATIONS = "nr_iterations"
GRAD_NORM_CLIP = "grad_norm_clip"
LEARNING_RATE = "learning_rate"
CLIP_RATIO = "clip_ratio"
UPDATE_ITERATIONS = "update_iterations"
VDN_MODE = "vdn_mode"
OUTPUT_DIM = "output_dim"
REWARD_SHARING = "reward_sharing"
SAMPLE_NR_AGENTS = "sample_nr_agents"
MIXING_HIDDEN_SIZE = "mixing_hidden_size"
NR_ATTENTION_HEADS = "nr_attention_heads"

# Curriculum Constants
CURRICULUM_NAME = "curriculum_name"
RANDOM_CURRICULUM = "Random"
CACTUS_CURRICULUM = "CACTUS"
RADIUS_UPDATE_INTERVAL = "radius_update_interval"
RESET_CURRICULUM_BUFFER = "reset_curriculum_buffer"
SUCCESS_RATE = "success_rate"
SUCCESS_RATE_VARIANCE = "success_rate_variance"
DEVIATION_FACTOR = "deviation_factor"
SLIDING_WINDOW_SIZE = "sliding_window_size"
IMPROVEMENT_THRESHOLD = "improvement_threshold"
