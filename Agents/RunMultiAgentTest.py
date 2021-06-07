import argparse
import math
import random
import time
import warnings

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from Utils.LevelSelection import LevelSelectionSchema
from LearningAgents.LearningAgentThread import MultiThreadTrajCollection
from SBEnviornment.SBEnvironmentWrapper import SBEnvironmentWrapper
from Utils.Config import config
from Utils.Parameters import Parameters

warnings.filterwarnings('ignore')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def sample_levels(training_level_set, num_agents, agent_idx):
    '''
    given idx, return the averaged distributed levels
    '''
    n = math.ceil(len(training_level_set) / num_agents)
    total_list = []
    for i in range(0, len(training_level_set), n):
        total_list.append(training_level_set[i:i + n])
    if agent_idx >= len(total_list):
        return None
    return total_list[agent_idx]


capabiilty_settings = {'1_01': {'train': ['1_01_01', '1_01_02', '1_01_03'], 'test': ['1_01_04', '1_01_05', '1_01_06']},
                       '1_02': {'train': ['1_02_01', '1_02_03', '1_02_04'], 'test': ['1_02_05', '1_02_06']},
                       '2_01': {'train': ['2_01_01', '2_01_02', '2_01_03', '2_01_04', '2_01_05'],
                                'test': ['2_01_06', '2_01_07', '2_01_08', '2_01_09']},
                       '2_02': {'train': ['2_02_01', '2_02_02', '2_02_03', '2_02_04'],
                                'test': ['2_02_05', '2_02_06', '2_02_07', '2_02_08']},
                       '2_03': {'train': ['2_03_01', '2_03_02', '2_03_03'], 'test': ['2_03_04', '2_03_05']},
                       '2_04': {'train': ['2_04_04', '2_04_05', '2_04_06'], 'test': ['2_04_02', '2_04_03']},
                       '3_01': {'train': ['3_01_01', '3_01_02', '3_01_03'], 'test': ['3_01_04', '3_01_06']},
                       '3_02': {'train': ['3_02_01', '3_02_02'], 'test': ['3_02_03', '3_02_04']},
                       '3_03': {'train': ['3_03_01', '3_03_02'], 'test': ['3_03_03', '3_03_04']},
                       '3_04': {'train': ['3_04_01', '3_04_02'], 'test': ['3_04_03', '3_04_04']},
                       '3_05': {'train': ['3_05_03', '3_05_04'], 'test': ['3_05_05']},
                       '3_06': {'train': ['3_06_01', '3_06_04', '3_06_06'],
                                'test': ['3_06_03', '3_06_05']},
                       '3_07': {'train': ['3_07_01', '3_07_02', '3_07_03'], 'test': ['3_07_04', '3_07_05']},
                       '3_08': {'train': ['3_08_01'], 'test': ['3_08_02']},
                       '3_09': {'train': ['3_09_01', '3_09_02', '3_09_03', '3_09_04'],
                                'test': ['3_09_07', '3_09_08']}}

if __name__ == '__main__':
    mode = 'cross'
    online_learning = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', metavar='N', type=str)
    parser.add_argument('--online_training', type=str2bool, default=True)
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()

    if args.mode == 'within':
        param = Parameters([args.template], args.online_training)
        c = config(**param.param)
        param_name = args.template

    elif args.mode == 'cross':
        capability_idx = args.template
        train_template = capabiilty_settings[capability_idx]['train']
        test_template = capabiilty_settings[capability_idx]['test']
        param = Parameters(template=train_template, if_online_learning=args.online_training,
                           test_template=test_template)
        c = config(**param.param)
        param_name = "capability_{}_cross".format(train_template[0][:4])

    param_name = param_name + "_" + c.network.__name__ + "_" + c.multiagent.__name__
    print('running {} mode template {} on {}'.format(args.mode, args.template, param_name))
    writer = SummaryWriter(log_dir='final_run/{}'.format(param_name), comment=param_name)
    network = c.network(c.h, c.w, c.output, writer, c.device).to(c.device)

    memory = c.memory_type(c.memory_size)
    optimizer = optim.Adam

    episodic_rewards = []
    winning_rates = []
    total_train_time = 0
    # we want the last 1 steps to be only 0.01, so the it should be eps_start*(eps_decay)**num_update_steps = 0.01
    eps_decay = (0.01 / c.eps_start) ** (1 / ((c.num_update_steps - 2)))  # leaving 2 steps to fully exploit
    level_winning_rate = {i: 0 for i in c.train_level_list}
    start_time = time.time()

    #####################################training phase#########################################
    for step in range(c.num_update_steps):
        c.train_time_per_ep = int(c.train_time_per_ep * c.train_time_rise)
        c.eps_start = c.eps_start * eps_decay
        ## using multi-threading to collect memory ##
        agents = []
        for i in range(c.num_worker):
            level_sampled = random.sample(c.train_level_list, c.num_level_per_agent)
            env = SBEnvironmentWrapper(reward_type=c.reward_type, speed=c.simulation_speed)
            agent = c.multiagent(id=i + 1, dqn=network, level_list=level_sampled, replay_memory=memory, env=env,
                                 level_selection_function=LevelSelectionSchema.RepeatPlay(
                                     c.training_attempts_per_level).select,
                                 EPS_START=c.eps_start, writer=writer)
            agents.append(agent)

        am = MultiThreadTrajCollection(agents)
        am.connect_and_run_agents()
        env.close()
        time.sleep(5)

        ## evaluate the agent's learning performance ##
        episodic_reward = []
        winning_rate = []
        max_reward = []
        max_winning_rate = []

        for idx in c.train_level_list:
            for agent in agents:
                try:
                    if idx in agent.level_list:
                        episodic_reward.append(np.average(agent.episode_rewards[idx]))
                        winning_rate.append(np.average(agent.did_win[idx]))
                        max_reward.append(np.max(agent.episode_rewards[idx]))
                        max_winning_rate.append(np.max(agent.did_win[idx]))
                        if int(level_winning_rate[idx]) < max(
                                agent.did_win[idx]):
                            level_winning_rate[idx] = max(
                                agent.did_win[idx])
                except KeyError:  # agent skipped level
                    episodic_reward.append(0)
                    winning_rate.append(0)

        writer.add_scalar("average_episodic_rewards", np.average(episodic_reward), memory.action_num)
        writer.add_scalar("average_winning_rates", np.average(winning_rate), memory.action_num)
        writer.add_scalar("max_episodic_rewards", np.average(max_reward), memory.action_num)
        writer.add_scalar(
            "max_winning_rates - training level is solved per agent",
            np.average(max_winning_rate), memory.action_num)
        # percent of task solved
        percent_task_solved = np.average(list(level_winning_rate.values()))
        writer.add_scalar("percent of training tasks solved",
                          percent_task_solved, memory.action_num)
        # del model and agents to free memory
        # del dqn
        del agents
        torch.cuda.empty_cache()

        ## training the network ##
        target_net = c.network(h=c.h, w=c.w, outputs=c.output, device=c.device).to(c.device)
        target_net.load_state_dict(network.state_dict())
        target_net.eval()
        network.train_model(target_net, total_train_time=total_train_time, train_time=c.train_time_per_ep,
                            train_batch=c.train_batch, gamma=c.gamma, memory=memory, optimizer=optimizer,
                            sample_eps=c.eps_start)

        del target_net
        torch.cuda.empty_cache()
        print('finish {} step'.format(step))
        end_time = time.time()
        print("running time: {:.2f}".format((end_time - start_time) / 60))
        total_train_time += c.train_time_per_ep

    print('training done')
    # training done, save the model
    network.save_model("LearningAgents/saved_model/{}_{}.pt".format(param_name, c.num_update_steps))
    del env

    #####################################testing phase#########################################
    total_train_time = 0
    memory = c.memory_type(c.memory_size)
    episodic_rewards = []
    winning_rates = []
    eps_decay = (0.01 / c.eps_start) ** (1 / c.test_steps)
    level_winning_rate = {i: 0 for i in c.test_level_list}
    test_train_time = c.train_time_per_ep // 4
    if not c.online_training:
        c.test_steps = 1
        c.eps_start = 0
    c.eps_start = c.eps_test
    test_attempts_per_level = 5
    for attempt in range(c.test_steps):
        c.eps_start = c.eps_start * eps_decay
        ## using multi-threading to collect memory ##
        ## 5 attempts per level per attempts to speed up evaluation ##
        agents = []
        for i in range(c.num_worker):
            level_sampled = sample_levels(c.test_level_list, c.num_worker, i)
            if not level_sampled:
                continue
            env = SBEnvironmentWrapper(reward_type=c.reward_type, speed=c.simulation_speed)
            agent = c.multiagent(id=i + 1, dqn=network, level_list=level_sampled, replay_memory=memory, env=env,
                                 level_selection_function=LevelSelectionSchema.RepeatPlay(
                                     test_attempts_per_level).select,
                                 EPS_START=c.eps_start, writer=writer)
            agent.action_selection_mode = 'sample'
            agents.append(agent)

        am = MultiThreadTrajCollection(agents)
        am.connect_and_run_agents()
        env.close()
        time.sleep(5)

        ## evaluate the agent's learning in testing performance ##
        episodic_reward = []
        winning_rate = []
        max_reward = []
        max_winning_rate = []

        for idx in c.test_level_list:
            for agent in agents:
                try:
                    if idx in agent.level_list:
                        episodic_reward.append(np.average(agent.episode_rewards[idx]))
                        winning_rate.append(np.average(agent.did_win[idx]))
                        max_reward.append(np.max(agent.episode_rewards[idx]))
                        max_winning_rate.append(np.max(agent.did_win[idx]))
                        if int(level_winning_rate[idx]) < max(
                                agent.did_win[idx]):
                            level_winning_rate[idx] = max(
                                agent.did_win[idx])
                except KeyError:  # agent skipped level
                    episodic_reward.append(0)
                    winning_rate.append(0)

        writer.add_scalar("average_testing_episodic_rewards", np.average(episodic_reward),
                          (attempt + 1) * test_attempts_per_level)
        writer.add_scalar("average_testing_winning_rates", np.average(winning_rate),
                          (attempt + 1) * test_attempts_per_level)
        writer.add_scalar("max_testing_episodic_rewards", np.average(max_reward),
                          (attempt + 1) * test_attempts_per_level)
        writer.add_scalar(
            "max_testing_winning_rates - level is solved",
            np.average(max_winning_rate), (attempt + 1) * test_attempts_per_level)
        # percent of task solved
        percent_task_solved = np.average(list(level_winning_rate.values()))
        writer.add_scalar("percent of testing tasks solved",
                          percent_task_solved, (attempt + 1) * test_attempts_per_level)
        writer.flush()
        del agents
        torch.cuda.empty_cache()

        ## online training phase ##
        if c.online_training and attempt != c.test_steps - 1:
            target_net = c.network(h=c.h, w=c.w, outputs=c.output, device=c.device).to(c.device)
            target_net.load_state_dict(network.state_dict())
            target_net.eval()
            network.train_model(target_net, total_train_time=total_train_time, train_time=test_train_time,
                                train_batch=c.train_batch, gamma=c.gamma, memory=memory, optimizer=optimizer)

            del target_net
            torch.cuda.empty_cache()
            print('finish one testing epoch')
            end_time = time.time()
            print("running time: {:.2f}".format((end_time - start_time) / 60))
            total_train_time += test_train_time

    total_end_time = time.time()
    print("finish running, total running time: {:.2f} mins".format((total_end_time - start_time) / 60))
