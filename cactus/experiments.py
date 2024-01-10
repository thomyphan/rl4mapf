from os.path import join
from cactus.constants import *
import cactus.curriculum as curr
import cactus.data as data
import time
import random

def run_episode(envs, controller, params, training_mode=True, render_mode=False, env_index=None):
    if env_index is not None:
        env = envs[env_index]
    else:
        env = random.choice(envs)
    done = False
    time_step = 0
    observations = env.reset()
    vertex_collisions = env.float_zeros(env.nr_agents)
    edge_collisions = env.float_zeros(env.nr_agents)
    info = {ENV_COMPLETION_RATE : 0.0}
    while not done:
        joint_action = controller.joint_policy(observations)
        next_observations, rewards, terminated, truncated, info = env.step(joint_action)
        vertex_collisions += info[ENV_VERTEX_COLLISIONS].to(FLOAT_TYPE)
        edge_collisions += info[ENV_EDGE_COLLISIONS].to(FLOAT_TYPE)
        done = env.is_done_all()
        time_step += 1
        if training_mode:
            controller.update(observations, joint_action, rewards, terminated, truncated, done, info)
        if render_mode:
            env.render()
        observations = next_observations
    return {
        DISCOUNTED_RETURNS: env.discounted_returns,
        UNDISCOUNTED_RETURNS: env.undiscounted_returns,
        VERTEX_COLLISIONS: vertex_collisions,
        EDGE_COLLISIONS: edge_collisions,
        TERMINATED: env.is_terminated().all(),
        COMPLETION_RATE: info[ENV_COMPLETION_RATE]
    }

def run_episodes(nr_episodes, envs, controller, params, training_mode=True, render_mode=False):
    successes = 0.0
    completion_rate_sum = 0.0
    for _ in range(nr_episodes):
        result = run_episode(envs, controller, params, training_mode, render_mode)
        completion_rate_sum += result[COMPLETION_RATE]
        if result[TERMINATED]:
            successes += 1
    success_rate = successes/nr_episodes
    success_rate_variance = success_rate*(1.0 - success_rate)
    return {
        SUCCESS_RATE: success_rate,
        SUCCESS_RATE_VARIANCE: success_rate_variance,
        COMPLETION_RATE: completion_rate_sum/nr_episodes
    }

def test_run(envs, controller, params, render_mode=False):
    successes = 0.0
    completion_rate_sum = 0.0
    nr_test_envs = len(envs)*1.0
    for i, env in enumerate(envs):
        backup_init_radius = env.init_goal_radius
        env.set_init_goal_radius(params[TEST_INIT_GOAL_RADIUS])
        result = run_episode(envs, controller, params, False, render_mode, env_index=i)
        completion_rate_sum += result[COMPLETION_RATE]
        if result[TERMINATED]:
            successes += 1
        env.set_init_goal_radius(backup_init_radius)
    success_rate = successes/nr_test_envs
    success_rate_variance = success_rate*(1.0 - success_rate)
    return {
        SUCCESS_RATE: success_rate,
        SUCCESS_RATE_VARIANCE: success_rate_variance,
        COMPLETION_RATE: completion_rate_sum/nr_test_envs
    }

def run_training(envs, test_envs, controller, params):
    curriculum = curr.make(params)(envs, params)
    episodes_per_epoch = params[EPISODES_PER_EPOCH]
    success_rates = []
    completion_rates = []
    prev_total_time = 0
    total_time = 0
    training_times = []
    areas_under_curve_success = []
    areas_under_curve_completion = []
    training_result = {COMPLETION_RATE : 0, SUCCESS_RATE_VARIANCE : 0}
    for i in range(params[NUMBER_OF_EPOCHS]+1):
        start = time.time()
        curriculum.update_curriculum(training_result[COMPLETION_RATE], training_result[SUCCESS_RATE_VARIANCE])
        training_result = run_episodes(episodes_per_epoch, envs, controller, params, training_mode=True, render_mode=params[RENDER_MODE])
        end = time.time() - start
        total_time += end
        if i%params[EPOCH_LOG_INTERVAL] == 0:
            training_time = total_time - prev_total_time
            prev_total_time = total_time
            result = test_run(test_envs, controller, params)
            print(f"Finished epoch {i} ({params[ALGORITHM_NAME]}, {params[CURRICULUM_NAME]}, {params[MAP_NAME]}, {params[ENV_NR_AGENTS]} agents):")
            print(f"- Success rate: {result[SUCCESS_RATE]}")
            print(f"- Completion rate: {result[COMPLETION_RATE]}")
            print(f"- Time elapsed: {training_time} seconds")
            success_rates.append(float(result[SUCCESS_RATE]))
            completion_rates.append(float(result[COMPLETION_RATE]))
            areas_under_curve_success.append(float(result[SUCCESS_RATE]*training_time))
            areas_under_curve_completion.append(float(result[COMPLETION_RATE]*training_time))
            training_times.append(training_time)
        if i > 0 and i%2000 == 0:
            controller.save_model_weights(params[DIRECTORY])
            result = {TOTAL_TIME: total_time, TIME_PER_EPOCH: total_time*1.0/(i + 1.0), SUCCESS_RATE: success_rates, COMPLETION_RATE: completion_rates, AUC_COMPLETION: areas_under_curve_completion, AUC_SUCCESS: areas_under_curve_success, TRAINING_TIME: training_times}
            data.save_json(join(params[DIRECTORY], f"results_{i}.json"), result)
    result = {
        TOTAL_TIME: total_time,
        TIME_PER_EPOCH: total_time*1.0/params[NUMBER_OF_EPOCHS],
        SUCCESS_RATE: success_rates,
        COMPLETION_RATE: completion_rates,
        AUC_COMPLETION: areas_under_curve_completion,
        AUC_SUCCESS: areas_under_curve_success,
        TRAINING_TIME: training_times
    }
    if DIRECTORY in params:
        data.save_json(join(params[DIRECTORY], "results.json"), result)
        controller.save_model_weights(params[DIRECTORY])
    return result