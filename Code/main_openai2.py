# Time: 2019-11-05
# Author: Zachary 
# Name: MADDPG_torch
# File func: main func
import os

import time
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from arguments import parse_args
from replay_buffer import ReplayBuffer
from model import openai_actor, openai_critic
from hard_env2 import ArmEnv


#获取训练模型
#观测空间18，动作空间5
def get_trainers(env, num_adversaries, obs_shape_n, action_shape_n, arglist):
    """
    init the trainers or load the old model
    """
    actors_cur = [None for _ in range(env.n)]
    critics_cur = [None for _ in range(env.n)]
    actors_tar = [None for _ in range(env.n)]
    critics_tar = [None for _ in range(env.n)]
    optimizers_c = [None for _ in range(env.n)]
    optimizers_a = [None for _ in range(env.n)]
    #所有观测和动作数
    input_size_global = sum(obs_shape_n) + sum(action_shape_n)
    #[18,18,18],[5,5.5]

    if arglist.restore == True: # restore the model
        for idx in range(1):
            actors_cur[idx] = torch.load(arglist.old_model_name+'a_c_{}.pt'.format(idx))
            actors_tar[idx] = torch.load(arglist.old_model_name+'a_t_{}.pt'.format(idx))
            critics_cur[idx] = torch.load(arglist.old_model_name + 'c_c_{}.pt'.format(idx))
            critics_tar[idx] = torch.load(arglist.old_model_name + 'c_t_{}.pt'.format(idx))
            optimizers_a[idx] = optim.Adam(actors_cur[idx].parameters(), arglist.lr_a)
            optimizers_c[idx] = optim.Adam(critics_cur[idx].parameters(), arglist.lr_c)
    # Note: if you need load old model, there should be a procedure for juding if the trainers[idx] is None
    #全部用openai网络结构

    for i in range(env.n):
        actors_cur[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        critics_cur[i] = openai_critic(26*4, 2*4, arglist).to(arglist.device)
        actors_tar[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        critics_tar[i] = openai_critic(26*4, 2*4, arglist).to(arglist.device)
        optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), arglist.lr_a)
        optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)

    actors_tar = update_trainers(actors_cur, actors_tar, 1.0) # update the target par using the cur
    critics_tar = update_trainers(critics_cur, critics_tar, 1.0) # update the target par using the cur
    return actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c
#更新目标网络，不用改
def update_trainers(agents_cur, agents_tar, tao):
    """
    update the trainers_tar par using the trainers_cur
    This way is not the same as copy_, but the result is the same
    out:
    |agents_tar: the agents with new par updated towards agents_current
    """
    for agent_c, agent_t in zip(agents_cur, agents_tar):
        key_list = list(agent_c.state_dict().keys())
        state_dict_t = agent_t.state_dict()
        state_dict_c = agent_c.state_dict()
        for key in key_list:
            state_dict_t[key] = state_dict_c[key]*tao + \
                    (1-tao)*state_dict_t[key] 
        agent_t.load_state_dict(state_dict_t)
    return agents_tar
def figshow(x,y):
    fig = plt.figure()
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    # 获得绘制的句柄
    ax1 = fig.add_axes([left, bottom, width, height])
    ax1.plot(x, y)
    ax1.set_title('simple_spread')
    plt.show()
#训练网络，重点
def agents_train(arglist, game_step, update_cnt, memory, obs_size, action_size, \
                actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c):
    """ 
    use this func to make the "main" func clean
    par:
    |input: the data for training
    |output: the data for next update
    """
    #obs_size和action_size都是什么样的？？？
    # update all trainers, if not in display or benchmark mode
    if game_step > arglist.learning_start_step and \
        (game_step - arglist.learning_start_step) % arglist.learning_fre == 0:
        if update_cnt == 0: print('\r=start training ...'+' '*100)
        # update the target par using the cur
        update_cnt += 1

        # update every agent in different memory batch
        for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
            enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):

            if opt_c == None: continue # jump to the next model update

            # sample the experience
            #观察sample函数，根据主函数写我自己的环境
            _obs_n_o, _action_n, _rew_n, _obs_n_n, _done_n = memory.sample( \
                arglist.batch_size, agent_idx) # Note_The func is not the same as others

            #[[],[]]采样两个，就是二维的，其中每个维都有三个智能体的状态动作
            #采样两个，奖励是两个，done也是两个，奖励是全局奖励？？？？[x,x]
            #一个奖励一个done
            #只改_obs_n_o,_obs_n_n即可
            rew = torch.tensor(_rew_n, device=arglist.device, dtype=torch.float) # set the rew to gpu
            done_n = torch.tensor(~_done_n, dtype=torch.float, device=arglist.device) # set the rew to gpu
            #4个agent的动作，4*5=20
            new_c_obs = []
            new_cn_obs=[]
            new_action=[]
            n_agent=[]
            for index in range(len(_obs_n_o)):
                xx=[]
                xxx=[]
                xxxx=[]
                a=[]
                aa=[]
                aaa=[]
                #xx存放critic需要的数据
                x1 = _obs_n_o[index][obs_size[agent_idx][0]:obs_size[agent_idx][1]]
                a1= _action_n[index][action_size[agent_idx][0]:action_size[agent_idx][1]]
                """找到其他所有agent的位置保存在x3中"""
                x2=_obs_n_o[index]
                l=int(len(_obs_n_o[index])/26)

                x3=np.zeros(l*2,float)
                i=0
                j=0
                while i<l*2-1:
                    x3[i]=x2[j]
                    i=i+1
                    j=j+1
                    x3[i]=x2[j]
                    i=i+1
                    j=j+25
                pos=x1[0:2]
                r1 = x1[6:8]
                r2 = x1[8:10]
                r3 = x1[10:12]
                p1 = pos+r1
                p2 = pos + r2
                p3 = pos + r3
                #print(round(p1[0], 3))
                #减1防止溢出

                for j in range(len(x3)-1):
                    jj=j+1
                    if round(x3[j],3)==round(p1[0],3) and round(x3[jj],3)==round(
                            p1[1],3):
                        d=_obs_n_o[index][obs_size[int(j/2)][0]:obs_size[int(j/2)][1]]
                        f=_action_n[index][action_size[int(j/2)][0]:action_size[int(j/2)][1]]
                        xx=np.hstack((x1,d))
                        a=np.hstack((a1,f))
                for j in range(len(x3) - 1):
                    jj = j + 1
                    if round(x3[j], 3) == round(p2[0], 3) and round(x3[jj], 3) == round(
                            p2[1], 3):
                        ff= _action_n[index][action_size[int(j / 2)][0]:action_size[int(j / 2)][1]]
                        dd = _obs_n_o[index][obs_size[int(j/2)][0]:obs_size[int(j/2)][1]]
                        xxx = np.hstack((xx, dd))
                        aa = np.hstack((a, ff))
                for j in range(len(x3) - 1):
                    jj = j + 1
                    if round(x3[j], 3) == round(p3[0], 3) and round(x3[jj], 3) == round(
                            p3[1], 3):
                        fff = _action_n[index][action_size[int(j / 2)][0]:action_size[int(j / 2)][1]]
                        ddd= _obs_n_o[index][obs_size[int(j/2)][0]:obs_size[int(j/2)][1]]
                        xxxx = np.hstack((xxx, ddd))
                        aaa = np.hstack((aa, fff))
                new_c_obs.append(xxxx)
                new_action.append(aaa)
            new_act=np.array(new_action)
            new_obs=np.array(new_c_obs)

            for index in range(len(_obs_n_n)):
                xx=[]
                xxx=[]
                xxxx=[]
                agent=[]
                #xx存放critic需要的数据
                x1 = _obs_n_n[index][obs_size[agent_idx][0]:obs_size[agent_idx][1]]
                """找到其他所有agent的位置保存在x3中"""
                x2=_obs_n_n[index]
                agent.append(agent_idx)
                l=int(len(_obs_n_n[index])/26)
                x3=np.zeros(l*2,float)
                i=0
                j=0
                while i<l*2-1:
                    x3[i]=x2[j]
                    i=i+1
                    j=j+1
                    x3[i]=x2[j]
                    i=i+1
                    j=j+25
                pos=x1[0:2]
                r1 = x1[6:8]
                r2 = x1[8:10]
                r3 = x1[10:12]
                p1 = pos+r1
                p2 = pos + r2
                p3 = pos + r3
                #print(round(p1[0], 3))
                #减1防止溢出
                for j in range(len(x3)-1):
                    jj=j+1
                    if round(x3[j],3)==round(p1[0],3) and round(x3[jj],3)==round(
                            p1[1],3):
                        d=_obs_n_n[index][obs_size[int(j/2)][0]:obs_size[int(j/2)][1]]
                        xx=np.hstack((x1,d))
                        agent.append(int(j/2))
                for j in range(len(x3) - 1):
                    jj = j + 1
                    if round(x3[j], 3) == round(p2[0], 3) and round(x3[jj], 3) == round(
                            p2[1], 3):
                        dd = _obs_n_n[index][obs_size[int(j/2)][0]:obs_size[int(j/2)][1]]
                        xxx = np.hstack((xx, dd))
                        agent.append(int(j / 2))
                for j in range(len(x3) - 1):
                    jj = j + 1
                    if round(x3[j], 3) == round(p3[0], 3) and round(x3[jj], 3) == round(
                            p3[1], 3):
                        ddd= _obs_n_n[index][obs_size[int(j/2)][0]:obs_size[int(j/2)][1]]
                        xxxx = np.hstack((xxx, ddd))
                        agent.append(int(j / 2))
                new_cn_obs.append(xxxx)
                n_agent.append(agent)
            new_nobs=np.array(new_cn_obs)

            action_cur_o = torch.from_numpy(_action_n).to(arglist.device, torch.float)
            obs_n_o = torch.from_numpy(_obs_n_o).to(arglist.device, torch.float)
            obs_n_n = torch.from_numpy(_obs_n_n).to(arglist.device, torch.float)

            nobs_n_o = torch.from_numpy(new_obs).to(arglist.device, torch.float)
            nobs_n_n = torch.from_numpy(new_nobs).to(arglist.device, torch.float)
            new_act = torch.from_numpy(new_act).to(arglist.device, torch.float)

            lk=[]
            for index in range(len(n_agent)):
                oo = []
                for i in range(len(n_agent[index])):
                    k=0
                    ee=actors_tar[n_agent[index][i]](nobs_n_n[index][k:k+26])
                    k+=26
                    oo.append(ee.detach().numpy()[0])
                    oo.append(ee.detach().numpy()[1])
                lk.append(oo)
            #ajk为输出的动作
            ajk=torch.from_numpy(np.array(lk)).to(arglist.device, torch.float)


            q = critic_c(nobs_n_o, new_act).reshape(-1) # q
            q_ = critic_t(nobs_n_n, ajk).reshape(-1) # q_
            tar_value = q_*arglist.gamma*done_n + rew # q_*gamma*done + reward
            loss_c = torch.nn.MSELoss()(q, tar_value) # bellman equation
            opt_c.zero_grad()
            loss_c.backward()
            nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)
            opt_c.step()
            #critic更新全部的，actor只用当前的obs_n_o
            # --use the data to update the ACTOR
            # There is no need to cal other agent's action
            model_out, policy_c_new = actor_c( \
                obs_n_o[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]], model_original_out=True)
            # update the aciton of this agent
            action_cur_o[:, action_size[agent_idx][0]:action_size[agent_idx][1]] = policy_c_new 
            loss_pse = torch.mean(torch.pow(model_out, 2))

            loss_a = torch.mul(-1, torch.mean(critic_c(nobs_n_o, new_act)))

            opt_a.zero_grad()
            (1e-3*loss_pse+loss_a).backward()
            nn.utils.clip_grad_norm_(actor_c.parameters(), arglist.max_grad_norm)
            opt_a.step()

        # save the model to the path_dir ---cnt by update number
        if update_cnt > arglist.start_save_model and update_cnt % arglist.fre4save_model == 0:
            time_now = time.strftime('%y%m_%d%H%M')
            print('=time:{} step:{}        save'.format(time_now, game_step))
            model_file_dir = os.path.join(arglist.save_dir, '{}_{}_{}'.format( \
                arglist.scenario_name, time_now, game_step))
            if not os.path.exists(model_file_dir): # make the path
                os.mkdir(model_file_dir)
            for agent_idx, (a_c, a_t, c_c, c_t) in \
                enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar)):
                torch.save(a_c, os.path.join(model_file_dir, 'a_c_{}.pt'.format(agent_idx)))
                torch.save(a_t, os.path.join(model_file_dir, 'a_t_{}.pt'.format(agent_idx)))
                torch.save(c_c, os.path.join(model_file_dir, 'c_c_{}.pt'.format(agent_idx)))
                torch.save(c_t, os.path.join(model_file_dir, 'c_t_{}.pt'.format(agent_idx)))

        # update the tar par
        actors_tar = update_trainers(actors_cur, actors_tar, arglist.tao) 
        critics_tar = update_trainers(critics_cur, critics_tar, arglist.tao) 

    return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar

def train(arglist):
    """
    init the env, agent and train the agents
    """
    """step1: create the environment """
    x=[]
    y=[]
    """生成一个env对象，调用函数"""
    env = ArmEnv(5)

    print('=============================')
    print('=1 Env {} is right ...'.format(arglist.scenario_name))
    print('=============================')
    obs_shape_n=[]
    action_shape_n=[]
    """step2: create agents"""
    for i in range(env.n):
        obs_shape_n.append(26)
        action_shape_n.append(2)
    #[18,18,18]
    #[5,5,5,5]torch.mean(critic_c(obs_n_o, action_cur_o))
    num_adversaries = min(env.n, arglist.num_adversaries)
    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = \
        get_trainers(env, num_adversaries, obs_shape_n, action_shape_n, arglist)
    #memory = Memory(num_adversaries, arglist)
    memory = ReplayBuffer(arglist.memory_size)
    
    print('=2 The {} agents are inited ...'.format(env.n))
    print('=============================')

    """step3: init the pars """
    obs_size = []
    action_size = []
    game_step = 0
    episode_cnt = 0
    update_cnt = 0
    t_start = time.time()
    rew_n_old = [0.0 for _ in range(env.n)] # set the init reward
    agent_info = [[[]]] # placeholder for benchmarking info
    episode_rewards = [0.0] # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)] # individual agent reward
    head_o, head_a, end_o, end_a = 0, 0, 0, 0
    for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
        end_o = end_o + obs_shape
        end_a = end_a + action_shape 
        range_o = (head_o, end_o)
        range_a = (head_a, end_a)
        obs_size.append(range_o)
        action_size.append(range_a)
        head_o = end_o
        head_a = end_a


    print('=3 starting iterations ...')
    print('=============================')
    obs_n = env.reset()

    for episode_gone in range(arglist.max_episode):
        # cal the reward print the debug data

        if game_step > 1 and game_step % 100 == 0:   
            mean_agents_r = [round(np.mean(agent_rewards[idx][-200:-1]), 2) for idx in range(env.n)]
            mean_ep_r = round(np.mean(episode_rewards[-200:-1]), 3)
            print(" "*43 + 'episode reward:{} agents mean reward:{}'.format(mean_ep_r, mean_agents_r))
            x.append(episode_gone)
            y.append(mean_ep_r)
        print('=Training: steps:{} episode:{}'.format(game_step, episode_gone), end='\r')
        #45步
        for episode_cnt in range(arglist.per_episode_max_len):
            # get action
            action_n = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                for agent, obs in zip(actors_cur, obs_n)]

            #  [array([0.03621321, 0.02920061, 0.7888186 , 0.14142624, 0.0043414 ],
            #       dtype=float32), array([0.7095107 , 0.0073576 , 0.1367599 , 0.13352096, 0.0128509 ],
            #       dtype=float32), array([0.02420491, 0.01804581, 0.6956599 , 0.19121987, 0.07086948],
            #       dtype=float32), array([0.00150357, 0.00411184, 0.5740526 , 0.0206397 , 0.3996924 ],
            #       dtype=float32)]

            # interact with env
            new_obs_n, rew_n, done_n= env.step(action_n)
            #env.render()
            """
            三个状态数组[[],[],[]]，三个奖励变量，三个done[,,]元组！！！
            """




            # save the experience
            #memory的add和sample要做一定的更改
            memory.add(obs_n, np.concatenate(action_n), rew_n , new_obs_n, done_n)
            episode_rewards[-1] += np.sum(rew_n)
            for i, rew in enumerate(rew_n): agent_rewards[i][-1] += rew

            # train our agents 
            update_cnt, actors_cur, actors_tar, critics_cur, critics_tar = agents_train(\
                arglist, game_step, update_cnt, memory, obs_size, action_size, \
                actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)

            # update the obs_n
            game_step += 1
            obs_n = new_obs_n
            done = all(done_n)
            terminal = (episode_cnt >= arglist.per_episode_max_len-1)

            if done or terminal :
                episode_step = 0
                obs_n = env.reset()
                episode_rewards.append(0)
                for a_r in agent_rewards:   
                    a_r.append(0)
                continue
    return x,y

if __name__ == '__main__':
    arglist = parse_args()
    xx,yy=train(arglist)
    print(xx)
    print(yy)
    figshow(xx,yy)


