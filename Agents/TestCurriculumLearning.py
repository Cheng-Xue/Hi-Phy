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
from LearningAgents.RLNetwork.DQNSymbolicDuelingFC_v2 import DQNSymbolicDuelingFC_v2

level_setting = {
    'level_1': ['1_01_01', '1_01_02', '1_01_03', '1_01_04', '1_01_05', '1_01_06', '1_02_01', '1_02_03',
                '1_02_04', '1_02_05', '1_02_06'],
    'level_2': ['2_01_01', '2_01_02', '2_01_03', '2_01_04', '2_01_05', '2_01_06', '2_01_07', '2_01_08', '2_01_09',
                '2_02_01', '2_02_02', '2_02_03', '2_02_04', '2_02_05', '2_02_06', '2_02_07', '2_02_08',
                '2_03_01', '2_03_02', '2_03_03', '2_03_04', '2_03_05', '2_04_02', '2_04_03', '2_04_04',
                '2_04_05', '2_04_06'],
    'level_3': ['3_01_01', '3_01_02', '3_01_03', '3_01_04', '3_01_06', '3_02_01', '3_02_02', '3_02_03', '3_02_04',
                '3_03_01', '3_03_02', '3_03_03', '3_03_04', '3_04_01', '3_04_02',
                '3_04_03', '3_04_04', '3_05_03', '3_05_04', '3_05_05', '3_06_01', '3_06_02', '3_06_03', '3_06_04',
                '3_06_05', '3_06_06', '3_06_07', '3_07_01', '3_07_02', '3_07_03',
                '3_07_04', '3_07_05', '3_09_01', '3_09_02', '3_09_03', '3_09_04', '3_09_07', '3_09_08']}

if __name__ == '__main__':
    mode = 'curriculum'
    online_learning = True
    network = None
    if mode == 'curriculum' or mode == 'reverse_curriculum' or mode == 'no_curriculum':
        # run 10 steps in the first level, then 10 in the second then 10 in the third
        writer = SummaryWriter(log_dir='final_run/{}'.format(mode), comment=mode)

        curriculum = None

        if mode == 'curriculum':
            curriculum = ['level_1', 'level_2', 'level_3']
        elif mode == 'reverse_curriculum':
            curriculum = ['level_3', 'level_2', 'level_1']

        if curriculum:
            for i, level in enumerate(curriculum):
                param = Parameters(level_setting[level], False)
                c = config(**param.param)
                if not network:
                    network = c.network(c.h, c.w, c.output, writer, c.device).to(c.device)
                    memory = c.memory_type(c.memory_size)
                    optimizer = optim.Adam

                total_train_time = 0
                # in curriculum learning, the exploration is decreased by /2 /3 in level 2 and level 3
                eps_decay = (0.01 / (c.eps_start / (i + 1))) ** (
                        1 / ((c.num_update_steps - 2)))  # leaving 2 steps to fully exploit
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
                        agent = c.multiagent(id=i + 1, dqn=network, level_list=level_sampled, replay_memory=memory,
                                             env=env,
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
                            except IndexError:  # agent skipped level
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


        elif mode == 'no_curriculum':
            all_template = []
            for key, value in level_setting.items():
                all_template.extend(value)
            param = Parameters(all_template, False)
            c = config(**param.param)
            if not network:
                network = c.network(c.h, c.w, c.output, writer, c.device).to(c.device)
                memory = c.memory_type(c.memory_size)
                optimizer = optim.Adam

            total_train_time = 0
            eps_decay = (0.01 / (c.eps_start)) ** (
                    1 / ((c.num_update_steps - 2)))  # leaving 2 steps to fully exploit
            level_winning_rate = {i: 0 for i in c.train_level_list}
            start_time = time.time()
            c.num_update_steps = c.num_update_steps * 3
            #####################################training phase#########################################
            for step in range(c.num_update_steps):
                c.train_time_per_ep = int(c.train_time_per_ep * c.train_time_rise)
                c.eps_start = c.eps_start * eps_decay
                ## using multi-threading to collect memory ##
                agents = []
                for i in range(c.num_worker):
                    level_sampled = random.sample(c.train_level_list, c.num_level_per_agent)
                    env = SBEnvironmentWrapper(reward_type=c.reward_type, speed=c.simulation_speed)
                    agent = c.multiagent(id=i + 1, dqn=network, level_list=level_sampled, replay_memory=memory,
                                         env=env,
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
                        except IndexError:  # agent skipped level
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
