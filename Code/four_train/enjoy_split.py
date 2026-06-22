import os
import sys
from mofang_env import ArmEnv
import torch
import numpy as np

from model import actor_agent, critic_agent
from arguments import parse_args



def get_trainers(env, arglist):
    trainers_cur = []
    trainers_tar = []
    optimizers = []
    input_size = [8, 10, 10] # the obs size
    input_size_global = [23, 25, 25] # cal by README

    """ load the model """
    actors_tars=[None for _ in range(1)]
    actors_tars1 = [None for _ in range(4)]
    #actors_tar = [torch.load(arglist.old_model_name+'a__{}.pt'.format(agent_idx), map_location=arglist.device) \
     #   for agent_idx in range(env.n)]
    for idx in range(1):
            actors_tars[idx] = torch.load(arglist.old_model_name+'a_t_{}.pt'.format(idx))

    return actors_tars


def enjoy(arglist):
    """ 
    This func is used for testing the model
    """
    actors_tar = [None for _ in range(8)]
    episode_step = 0
    """ init the env """
    env = ArmEnv(1)

    """ init the agents """
   # obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    actors_tar = get_trainers(env, arglist)
    """ interact with the env """
    obs_n = env.reset()
    x=[]
    y=[]
    while(1):

        x.append(obs_n[0][0]*1000)
        y.append(obs_n[0][1]* 1000)
        # update the episode step number
        episode_step += 1

        # get action
     #   try:
        action_n = []
            # action_n = [agent.actor(torch.from_numpy(obs).to(arglist.device, torch.float)).numpy() \
            # for agent, obs in zip(trainers_cur, obs_n)]


        action_n = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                    for agent, obs in zip(actors_tar, obs_n)]


#        except:
#           print(obs_n)

        #env.render()
        # interact with env
        new_obs_n, rew_n, done_n= env.step(action_n)

        # update the flag
        done = all(done_n)
        terminal = (episode_step >= 200)
        obs_n=new_obs_n
        # reset the env
        if done or terminal:

            np.savetxt('l1.txt', x, fmt='%0.2f')
            np.savetxt('l2.txt', y, fmt='%0.2f')
            episode_step = 0
            break
            obs_n = env.reset()

        # render the env
        #print(rew_n)
        env.render()

if __name__ == '__main__':
    arglist = parse_args()
    enjoy(arglist)
