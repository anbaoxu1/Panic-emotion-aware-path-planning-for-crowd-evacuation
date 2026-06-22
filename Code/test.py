import os
import sys
from test_env3 import ArmEnv
import torch


from model import actor_agent, critic_agent
from arguments import parse_args



def get_trainers(env, arglist):
    trainers_cur = []
    trainers_tar = []
    optimizers = []
    input_size = [8, 10, 10] # the obs size
    input_size_global = [23, 25, 25] # cal by README

    """ load the model """
    actors_tars=[None for _ in range(32)]

    #actors_tar = [torch.load(arglist.old_model_name+'a__{}.pt'.format(agent_idx), map_location=arglist.device) \
     #   for agent_idx in range(env.n)]
    for idx in range(32):
           # print(idx)

        actors_tars[idx] = torch.load(arglist.old_model_name+'a_t_{}.pt'.format(0))
        """
            else:
                actors_tars[idx] = torch.load(arglist.old_model_name + 'a_t_{}.pt'.format(idx-8))
        """
    return actors_tars


def enjoy(arglist):
    """ 
    This func is used for testing the model
    """
    actors_tar = [None for _ in range(8)]
    episode_step = 0
    """ init the env """
    env = ArmEnv(32)

    """ init the agents """
   # obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    actors_tar = get_trainers(env, arglist)
    """ interact with the env """
    obs_n = env.reset()

    while(1):

        # update the episode step number
        episode_step += 1

        # get action
     #   try:
        action_n = []
            # action_n = [agent.actor(torch.from_numpy(obs).to(arglist.device, torch.float)).numpy() \
            # for agent, obs in zip(trainers_cur, obs_n)]
        for actor, obs in zip(actors_tar, obs_n):
                #action = torch.clamp(actor(torch.from_numpy(obs).to(arglist.device, torch.float)), -1, 1)
                #action_n.append(action)

            action_n = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                    for agent, obs in zip(actors_tar, obs_n)]


#        except:
#           print(obs_n)

        #env.render()
        # interact with env
        new_obs_n, rew_n, done_n,l= env.step(action_n)

        # update the flag
        done = all(done_n)
        terminal = (episode_step >= 100)
        obs_n=new_obs_n
        # reset the env
        if done or terminal: 
            episode_step = 0
            obs_n = env.reset()

        # render the env
        #print(rew_n)
        env.render()

if __name__ == '__main__':
    arglist = parse_args()
    enjoy(arglist)
