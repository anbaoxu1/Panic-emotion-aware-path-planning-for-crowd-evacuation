import os
import sys
from suiji_env import ArmEnv
import torch
import matplotlib.pyplot as plt

from model import actor_agent, critic_agent
from arguments import parse_args

import time
num=0
def get_trainers(env, arglist):
    trainers_cur = []
    trainers_tar = []
    optimizers = []
    input_size = [8, 10, 10] # the obs size
    input_size_global = [23, 25, 25] # cal by README

    """ load the model """
    actors_tars=[None for _ in range(1)]

    #actors_tar = [torch.load(arglist.old_model_name+'a__{}.pt'.format(agent_idx), map_location=arglist.device) \
     #   for agent_idx in range(env.n)]
    actors_tars[0] = torch.load(arglist.old_model_name+'a_t_{}.pt'.format(0))
    return actors_tars

def figshow(x,y):
    fig = plt.figure()
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    # 获得绘制的句柄
    ax1 = fig.add_axes([left, bottom, width, height])
    ax1.plot(x, y)
    ax1.set_title('my_trac')
    plt.show()
def enjoy(arglist):
    """ 
    This func is used for testing the model
    """

    episode_step = 0
    """ init the env """
    env = ArmEnv(1)
    """ init the agents """
   # obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    actors_tar = get_trainers(env, arglist)
    """ interact with the env """
    obs_n = env.reset()
    num=0
    while(1):

        # update the episode step number
        episode_step += 1
        # get action
     #   try:
        action_n = []
        start = time.perf_counter()
        action_n = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                for agent, obs in zip(actors_tar, obs_n)]
#        except:
#           print(obs_n)
        end = time.perf_counter()


        # interact with env
        new_obs_n, rew_n, done_n,x= env.step(action_n)
        env.render()
        # update the flag
        done = all(done_n)
        terminal = (episode_step >= 100)
        obs_n=new_obs_n
        # reset the env
        if done or terminal or x:

            episode_step = 0
            obs_n = env.reset()

        # render the env
        #print(rew_n)

if __name__ == '__main__':
    arglist = parse_args()
    start = time.perf_counter()
    enjoy(arglist)
    end = time.perf_counter()
    print(end-start)