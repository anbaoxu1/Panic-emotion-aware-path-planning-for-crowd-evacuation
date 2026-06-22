import pyglet
import numpy as np
import copy
from RVO import RVO_update, reach, compute_V_des, reach
import math
import matplotlib.pyplot as plt
import time
from itertools import combinations

import operator

from functools import reduce


#reset,step,reward

"""
2020.10.2日重写代码

"""
class Agent():
    def __init__(self,pos,vel,goal):
        self.p=pos
        self.v=vel
        self.g=goal




    def setatt(self,a,b,c):
        self.p=a
        self.v=b
        self.g=c



class Viewer(pyglet.window.Window):
    #传一个agent数组即可
    def __init__(self,agentinfo):
        ...
        # 初始化位置数组和目标数组
        self.p=[]
        self.g=[]
        #画图数组
        self.pos=[]
        self.goal=[]
        for info in agentinfo:
            self.p.append(info.p)
            self.g.append(info.g)

        # 创建窗口的继承
        # vsync 如果是 True, 按屏幕频率刷新, 反之不按那个频率
        super(Viewer, self).__init__(width=800, height=800, resizable=False, caption='Agent', vsync=False)
        # 窗口背景颜色
        pyglet.gl.glClearColor(1, 1, 1, 1)
        """
        半径大小
        """
        self.goalmax = 10
        self.agentmax = 20
        self.batch = pyglet.graphics.Batch()  # display whole batch at once
        for index in range(len(self.g)):
            temp=tuple(np.random.randint(0,255,3))
            self.goal.append(self.batch.add(
                4, pyglet.gl.GL_QUADS, None,  # 4 corners
                ('v2f', [self.g[index][0] - self.goalmax , self.g[index][1] - self.goalmax ,
                         self.g[index][0] - self.goalmax , self.g[index][1] + self.goalmax ,
                         self.g[index][0] + self.goalmax , self.g[index][1] + self.goalmax ,
                         self.g[index][0] + self.goalmax , self.g[index][1] - self.goalmax]),
                ('c3B', temp* 4),))
            self.pos.append(self.batch.add(
                4, pyglet.gl.GL_QUADS, None,  # 4 corners
                ('v2f', [self.p[index][0] - self.agentmax , self.p[index][1] - self.agentmax ,
                         self.p[index][0] - self.agentmax , self.p[index][1] + self.agentmax ,
                         self.p[index][0] + self.agentmax , self.p[index][1] + self.agentmax ,
                         self.p[index][0] + self.agentmax , self.p[index][1] - self.agentmax ]),
                ('c3B', temp * 4),))
        """
        kuan=5
        self.obs1=(self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [35-kuan, 400-380,
                     35-kuan, 400+380,
                     35+kuan, 400+380,
                     35+kuan, 400-380]),
            ('c3B', (0,0,255) * 4), ))
        self.obs2 = (self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [40, 770,
                     40, 780,
                     500,780,
                     500, 770]),
            ('c3B', (0, 0, 255) * 4), ))
        self.obs3 = (self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [650, 770,
                     650, 780,
                     790, 780,
                     790, 770]),
            ('c3B', (0, 0, 255) * 4), ))
        self.obs4 = (self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [780, 400,
                     780, 770,
                     790, 770,
                     790, 400]),
            ('c3B', (0, 0, 255) * 4), ))
        self.obs5 = (self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [780, 20,
                     780, 300,
                     790, 300,
                     790, 20]),
            ('c3B', (0, 0, 255) * 4), ))
        self.obs6 = (self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [40, 20,
                     40, 30,
                     780, 30,
                     780, 20]),
            ('c3B', (0, 0, 255) * 4), ))
        self.obs7 = (self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [40, 400,
                     40, 410,
                     400, 410,
                     400, 400]),
            ('c3B', (0, 0, 255) * 4), ))
        self.obs8 = (self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [400, 200,
                     400, 600,
                     410, 600,
                     410, 200]),
            ('c3B', (0, 0, 255) * 4), ))
        self.obs9 = (self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [400, 700,
                     400, 770,
                     410, 770,
                     410, 700]),
            ('c3B', (0, 0, 255) * 4), ))
        self.obs9 = (self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [400, 30,
                     400, 100,
                     410, 100,
                     410, 30]),
            ('c3B', (0, 0, 255) * 4), ))
        """
    def on_draw(self):
        self.clear()  # 清屏
        self.batch.draw()  # 画上 batch 里面的内容

    #更新agent的位置
    def _update_arm(self,p,g):
        for index in range(len(p)):
            a = (p[index][0] - self.agentmax, p[index][1] - self.agentmax)
            b = (p[index][0] - self.agentmax, p[index][1] + self.agentmax)
            c = (p[index][0]+ self.agentmax,  p[index][1] + self.agentmax)
            d = (p[index][0]+ self.agentmax,  p[index][1] - self.agentmax)
            self.pos[index].vertices = np.concatenate((a,b,c,d))
        for index in range(len(g)):
            a = (g[index][0] - self.goalmax, g[index][1] - self.goalmax)
            b = (g[index][0] - self.goalmax, g[index][1] + self.goalmax)
            c = (g[index][0]+ self.goalmax,  g[index][1] + self.goalmax)
            d = (g[index][0]+ self.goalmax,  g[index][1] - self.goalmax)
            self.goal[index].vertices = np.concatenate((a,b,c,d))
    def render(self,p,g):
        self._update_arm(p,g)
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()



class  ArmEnv(object):
    viewer = None  # 首先没有 viewer
    dt =20# refresh rate
    ws_model = dict()
    ws_model['robot_radius'] = 30
    ws_model['circular_obstacles'] = []
    ws_model['boundary'] = []
    def __init__(self,n):
        #初始化Agent
        self.n=n
        self.Agent=[]

        for i in range(n):
            x = np.random.randint(50,750,2)
            y = np.random.randint(50,750,2)
            agent = Agent(x, [0, 0], y)
            self.Agent.append(agent)
        """
        for i in range(n):
            if i%2==0:
                x = float(np.random.randint(50, 380, 1))
                y = float(np.random.randint(420, 750, 1))
                agent = Agent([x,y], [0, 0], [550,790], 10, 50)
            else:
                x = float(np.random.randint(40, 390, 1))
                y = float(np.random.randint(40, 390, 1))
                agent = Agent([x, y], [0, 0], [780, 350], 10, 50)
            self.Agent.append(agent)
        """
    def computerdis(self,x,y):
        return math.sqrt(np.square(x[0]-y[0])+np.square(x[1]-y[1]))

    def step(self,action):
        #action是列表里面套array数组
        #reward直接在step中定义即可
        dist=[]

        W1=0.6
        W2=0.4
        done=[]
        #初始化done元组
        for index in range(len(self.Agent)):
            done.append(False)
        reward=[]
        obs=[]
        #1.首先计算reward,先计算出RVO速度和当前速度即action（a1,a2）做对比
        P=np.zeros((self.n,2))
        for index in range(len(self.Agent)):
            P[index]=self.Agent[index].p
        V = np.zeros((self.n,2))
        for index in range(len(self.Agent)):
            V[index]=self.Agent[index].v
        G = np.zeros((self.n,2))
        for index in range(len(self.Agent)):
            G[index]=self.Agent[index].g
        V_max = [1.0 for i in range(len(self.Agent))]

        # 计算首选速度
        V_des = compute_V_des(P,G,V_max)
        # 计算RVO速度
        V = RVO_update(P, V_des, V, self.ws_model)
        pre_pos=np.zeros((self.n,2))
        #计算reward
        for index in range(len(action)):
            temp=np.sqrt(np.sum(np.square(action[index] - V[0])))
            reward.append(math.exp(-temp)*W1)
            pre_pos[index]=copy.deepcopy(self.Agent[index].p)
            self.Agent[index].p[0]+=action[index][0]*self.dt
            self.Agent[index].p[1]+=action[index][1]*self.dt
            self.Agent[index].v=action[index]
        #更新智能体状态,获得新obs
        ##计算reward，算出上一步和下一步的
        for index in range(len(self.Agent)):
            asd=self.length(pre_pos[index][0],pre_pos[index][1],
                            self.Agent[index].p[0],self.Agent[index].p[1],self.Agent[index].g[0],
                            self.Agent[index].g[1])*0.1

            reward[index]+=asd*W2

        """
        计算离当前智能体最近的3个智能体的状态（距离目标位置，相对距离，相对速度）
        """
        for index in range(len(self.Agent)):
            list1 = []
            for dis in range(len(self.Agent)):
                if dis != index:
                    list1.append((self.Agent[dis], self.dist(self.Agent[dis], self.Agent[index])))
            list1.sort(key=lambda x: x[1], reverse=False)

            relate_p = []
            relate_g = []
            relate_v = []
            pp = []
            gg = []
            vv = []
            relate_g.append(self.Agent[index].p - self.Agent[index].g)
            for i in range(len(list1)):
                if i <= 2:
                    relate_p.append(list1[i][0].p - self.Agent[index].p)
                    relate_v.append(np.array(list1[i][0].v) - np.array(self.Agent[index].v))
                    relate_g.append(list1[i][0].p - self.Agent[i].g)
            for b in relate_p:
                pp.append(b[0])
                pp.append(b[1])
            for g in relate_g:
                gg.append(g[0])
                gg.append(g[1])
            for v in relate_v:
                gg.append(v[0])
                gg.append(v[1])
            xobs = np.concatenate([np.array(
                [self.Agent[index].p[0], self.Agent[index].p[1], self.Agent[index].v[0], self.Agent[index].v[1],
                 self.Agent[index].g[0], self.Agent[index].g[1],
                 ])] + [pp] + [gg] + [vv]) * 0.001
            obs.append(xobs)


        for index in range(len(self.Agent)):
            if self.dist1(self.Agent[index]):
                reward[index] += 1
                done[index] = True
        x = combinations(self.Agent, 2)
        for test in x:
            if self.col(test[0],test[1],60):
                for index in range(len(self.Agent)):
                    if test[0] == self.Agent[index]:
                        reward[index]-=1
                    if test[1] == self.Agent[index]:
                        reward[index]-=1
        return obs,reward,done

    def dist(self,a1,a2):
        return np.sqrt(np.sum(np.square(a1.p - a2.p)))

    def dist1(self,a1):
        x= np.sqrt(np.sum(np.square(a1.p - a1.g)))
        if x<3:
            return True
        else:
            return False


    def get_obs(self):
        kk=[]
        for index in range(len(self.Agent)):
            list1 = []
            for dis in range(len(self.Agent)):
                if dis != index:
                    list1.append((self.Agent[dis], self.dist(self.Agent[dis], self.Agent[index])))
            list1.sort(key=lambda x: x[1], reverse=False)

            relate_p = []
            relate_g = []
            relate_v = []
            pp = []
            gg = []
            vv = []
            relate_g.append(self.Agent[index].p - self.Agent[index].g)
            for i in range(len(list1)):
                if i <= 2:
                    relate_p.append(list1[i][0].p - self.Agent[index].p)
                    relate_v.append(np.array(list1[i][0].v) - np.array(self.Agent[index].v))
                    relate_g.append(list1[i][0].p - list1[i][0].g)
            print(len(relate_g))
            for b in relate_p:
                pp.append(b[0])
                pp.append(b[1])
            for g in relate_g:
                gg.append(g[0])
                gg.append(g[1])
            for v in relate_v:
                gg.append(v[0])
                gg.append(v[1])
            xobs = np.concatenate([np.array(
                [self.Agent[index].p[0], self.Agent[index].p[1], self.Agent[index].v[0], self.Agent[index].v[1],
                 self.Agent[index].g[0], self.Agent[index].g[1],
                 ])] + [pp] + [gg] + [vv]) * 0.001
            kk.append(xobs)
        return len(kk[0])

#重新写reset环境，使得智能体出现在不同位置，并且不会与障碍物重叠
    def reset(self):
        obs=[]

        for index in range(len(self.Agent)):
            x = np.random.randint(50, 750, 2)
            y = np.random.randint(50, 750, 2)
            self.Agent[index].setatt(x,[0,0],y)
        """

        self.Agent[0].setatt(np.array([400,200]), [0, 0], np.array([400,600]))
        self.Agent[1].setatt(np.array([400, 600]), [0, 0], np.array([400, 200]))
        self.Agent[2].setatt(np.array([200, 400]), [0, 0], np.array([600, 400]))
        self.Agent[3].setatt(np.array([600, 400]), [0, 0], np.array([200, 400]))
       # self.Agent[4].setatt(np.array([600, 200]), [0, 0], np.array([200, 600]))
        """
        for index in range(len(self.Agent)):
            list1 = []
            for dis in range(len(self.Agent)):
                if dis != index:
                    list1.append((self.Agent[dis], self.dist(self.Agent[dis], self.Agent[index])))
            list1.sort(key=lambda x: x[1], reverse=False)

            relate_p = []
            relate_g = []
            relate_v = []
            pp = []
            gg = []
            vv = []
            relate_g.append(self.Agent[index].p - self.Agent[index].g)
            for i in range(len(list1)):
                if i <= 2:
                    relate_p.append(list1[i][0].p - self.Agent[index].p)
                    relate_v.append(np.array(list1[i][0].v) - np.array(self.Agent[index].v))
                    relate_g.append(list1[i][0].p - self.Agent[i].g)
            for b in relate_p:
                pp.append(b[0])
                pp.append(b[1])
            for g in relate_g:
                gg.append(g[0])
                gg.append(g[1])
            for v in relate_v:
                gg.append(v[0])
                gg.append(v[1])
            xobs = np.concatenate([np.array(
                [self.Agent[index].p[0], self.Agent[index].p[1], self.Agent[index].v[0], self.Agent[index].v[1],
                 self.Agent[index].g[0], self.Agent[index].g[1],
                 ])] + [pp] + [gg] + [vv]) * 0.001
            obs.append(xobs)
        return obs


    def test_RVO(self):

        P = np.zeros((self.n, 2))
        for index in range(len(self.Agent)):
            P[index] = self.Agent[index].p
        V = np.zeros((self.n, 2))
        for index in range(len(self.Agent)):
            V[index] = self.Agent[index].v
        G = np.zeros((self.n, 2))
        for index in range(len(self.Agent)):
            G[index] = self.Agent[index].g
        V_max = [1.0 for i in range(len(self.Agent))]

        # 计算首选速度
        V_des = compute_V_des(P, G, V_max)
        # 计算RVO速度
        V = RVO_update(P, V_des, V, self.ws_model)
        for index in range(len(V)):
            self.Agent[index].p[0]+= V[index][0]*self.dt
            self.Agent[index].p[1]+= V[index][1]*self.dt
            self.Agent[index].v[0] = V[index][0]
            self.Agent[index].v[1] = V[index][1]

    def render(self):
        if self.viewer is None:  # 如果调用了 render, 而且没有 viewer, 就生成一个
            self.viewer = Viewer(self.Agent)
        p=[]
        for index in range(len(self.Agent)):
            p.append(self.Agent[index].p)
        g = []
        for index in range(len(self.Agent)):
            g.append(self.Agent[index].g)
        self.viewer.render(p,g)  # 使用 Viewer 中的 render 功能

    def sample_action(self):
        return np.random.rand(2)*20-10  # two radians

    def collection(self,x1,y1,h1,x2,y2,h2):
        #如果碰撞，则返回真
        if abs(x1-x2)< h1/2+h2/2 and abs(y1-y2)<h1/2+h2/2:
            return True
        else:
            return False
    def col(self,agent1,agent2,R):
        if (np.sqrt((agent1.p[0]-agent2.p[0])**2+(agent1.p[1]-agent2.p[1])**2))<R:
            return True
        else:
            return False
    def length(self,x1,y1,x2,y2,ta1,ta2):
        temp=(np.sqrt((x1-ta1)**2+(y1-ta2)**2))-(np.sqrt((x2-ta1)**2+(y2-ta2)**2))
        return temp
    def judge(self,x1,y1,x2,y2):
        if x1==x2 and y1==y2:
            return True
        else:
            return False

    def figshow(self,x,y):
        fig = plt.figure()
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        # 获得绘制的句柄
        ax1 = fig.add_axes([left, bottom, width, height])
        ax1.plot(x, y)
        ax1.set_title('simple_spread')
        plt.show()
if __name__ == '__main__':
    e=ArmEnv(4)
    for a in e.Agent:
        print(a.p)
    a=[(e.Agent[0],3),(e.Agent[1],2),(e.Agent[2],1),(e.Agent[3],4)]
    a.sort(key = lambda x: x[1], reverse = False)
    for aj in a:
        print(aj[0].p)


