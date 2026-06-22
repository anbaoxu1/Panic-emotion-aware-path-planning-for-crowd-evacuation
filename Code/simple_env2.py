import pyglet
import numpy as np
import copy
from RVO import RVO_update, reach, compute_V_des, reach
import math
import matplotlib.pyplot as plt
import time
from itertools import combinations
#reset,step,reward
class Agent():
    def __init__(self,pos,vel,goal,radiu,gradiu):
        self.p=pos
        self.v=vel
        self.g=goal
        self.r=radiu
        self.gr=gradiu
    def setatt(self,a,b,c,d,e):
        self.p=a
        self.v=b
        self.g=c
        self.r=d
        self.gr=e
class Viewer(pyglet.window.Window):
    def __init__(self, agent_info1, agent_info2,agent_info3, agent_info4,agent_info5, agent_info6,agent_info7, agent_info8):
        ...
        # 添加 arm 信息

        self.p1=agent_info1.p
        self.g1=agent_info1.g
        self.p2 = agent_info2.p
        self.g2 = agent_info2.g
        self.p3 = agent_info3.p
        self.g3 = agent_info3.g
        self.p4 = agent_info4.p
        self.g4 = agent_info4.g

        self.p5 = agent_info5.p
        self.g5 = agent_info5.g
        self.p6 = agent_info6.p
        self.g6 = agent_info6.g
        self.p7 = agent_info7.p
        self.g7 = agent_info7.g
        self.p8 = agent_info8.p
        self.g8 = agent_info8.g
        #goal是一个三元组，x,y,l(长度)
        # 创建窗口的继承
        # vsync 如果是 True, 按屏幕频率刷新, 反之不按那个频率
        super(Viewer, self).__init__(width=800, height=800, resizable=False, caption='Agent', vsync=False)
        # 窗口背景颜色
        pyglet.gl.glClearColor(1, 1, 1, 1)

        # 将手臂的作图信息放入这个 batch
        self.batch = pyglet.graphics.Batch()  # display whole batch at once

        # 添加目标goal
        self.goal1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [self.g1[0] - 50 / 2, self.g1[1] - 50/ 2,
                     self.g1[0] - 50 / 2, self.g1[1] + 50 / 2,
                     self.g1[0] + 50/ 2, self.g1[1] + 50 / 2,
                     self.g1[0] + 50/ 2, self.g1[1] - 50  / 2]),
            ('c3B', (45, 45, 45) * 4))  # color
        self.goal2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [self.g2[0] - 50 / 2, self.g2[1] - 50 / 2,
                     self.g2[0] - 50 / 2, self.g2[1] + 50 / 2,
                     self.g2[0] + 50 / 2, self.g2[1] + 50 / 2,
                     self.g2[0] + 50 / 2, self.g2[1] - 50 / 2]),
            ('c3B', (99, 68, 156) * 4))  # color

        self.goal3 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [self.g3[0] - 50 / 2, self.g3[1] - 50 / 2,
                     self.g3[0] - 50 / 2, self.g3[1] + 50 / 2,
                     self.g3[0] + 50 / 2, self.g3[1] + 50 / 2,
                     self.g3[0] + 50 / 2, self.g3[1] - 50 / 2]),
            ('c3B', (15, 65, 156) * 4))  # color
        self.goal4 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [self.g4[0] - 50 / 2, self.g4[1] - 50 / 2,
                     self.g4[0] - 50 / 2, self.g4[1] + 50 / 2,
                     self.g4[0] + 50 / 2, self.g4[1] + 50 / 2,
                     self.g4[0] + 50 / 2, self.g4[1] - 50 / 2]),
            ('c3B', (245, 168, 156) * 4,))  # color
        self.goal5 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [self.g5[0] - 50 / 2, self.g5[1] - 50 / 2,
                     self.g5[0] - 50 / 2, self.g5[1] + 50 / 2,
                     self.g5[0] + 50 / 2, self.g5[1] + 50 / 2,
                     self.g5[0] + 50 / 2, self.g5[1] - 50 / 2]),
            ('c3B', (45, 45, 45) * 4))  # color
        self.goal6 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [self.g6[0] - 50 / 2, self.g6[1] - 50 / 2,
                     self.g6[0] - 50 / 2, self.g6[1] + 50 / 2,
                     self.g6[0] + 50 / 2, self.g6[1] + 50 / 2,
                     self.g6[0] + 50 / 2, self.g6[1] - 50 / 2]),
            ('c3B', (99, 68, 156) * 4))  # color

        self.goal7 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [self.g7[0] - 50 / 2, self.g7[1] - 50 / 2,
                     self.g7[0] - 50 / 2, self.g7[1] + 50 / 2,
                     self.g7[0] + 50 / 2, self.g7[1] + 50 / 2,
                     self.g7[0] + 50 / 2, self.g7[1] - 50 / 2]),
            ('c3B', (15, 65, 156) * 4))  # color
        self.goal8 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [self.g8[0] - 50 / 2, self.g8[1] - 50 / 2,
                     self.g8[0] - 50 / 2, self.g8[1] + 50 / 2,
                     self.g8[0] + 50 / 2, self.g8[1] + 50 / 2,
                     self.g8[0] + 50 / 2, self.g8[1] - 50 / 2]),
            ('c3B', (245, 168, 156) * 4,))  # color

        self.agent1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [self.p1[0]-30/2, self.p1[1]-30/2,  # location
                     self.p1[0]-30/2, self.p1[1]+30/2,
                     self.p1[0]+30/2, self.p1[1]+30/2,
                     self.p1[0]+30/2, self.p1[1]-30/2]), ('c3B', (45, 45, 45) * 4,))
        self.agent2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [self.p2[0] - 30 / 2, self.p2[1] - 30 / 2,  # location
                     self.p2[0] - 30 / 2, self.p2[1] + 30 / 2,
                     self.p2[0] + 30 / 2, self.p2[1] + 30 / 2,
                     self.p2[0] + 30 / 2, self.p2[1] - 30 / 2]), ('c3B', (99, 68, 156) * 4,))

        self.agent4= self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [self.p4[0] - 30 / 2, self.p4[1] - 30 / 2,  # location
                     self.p4[0] - 30 / 2, self.p4[1] + 30 / 2,
                     self.p4[0] + 30 / 2, self.p4[1] + 30 / 2,
                     self.p4[0] + 30 / 2, self.p4[1] - 30 / 2]), ('c3B', (245, 168, 156) * 4,))
        self.agent3 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [self.p3[0] - 30 / 2, self.p3[1] - 30 / 2,  # location
                     self.p3[0] - 30 / 2, self.p3[1] + 30 / 2,
                     self.p3[0] + 30 / 2, self.p3[1] + 30 / 2,
                     self.p3[0] + 30 / 2, self.p3[1] - 30 / 2]), ('c3B', (15, 65, 156) * 4,))
        self.agent5 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [self.p5[0] - 30 / 2, self.p5[1] - 30 / 2,  # location
                     self.p5[0] - 30 / 2, self.p5[1] + 30 / 2,
                     self.p5[0] + 30 / 2, self.p5[1] + 30 / 2,
                     self.p5[0] + 30 / 2, self.p5[1] - 30 / 2]), ('c3B', (45, 45, 45) * 4,))
        self.agent6= self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [self.p6[0] - 30 / 2, self.p6[1] - 30 / 2,  # location
                     self.p6[0] - 30 / 2, self.p6[1] + 30 / 2,
                     self.p6[0] + 30 / 2, self.p6[1] + 30 / 2,
                     self.p6[0] + 30 / 2, self.p6[1] - 30 / 2]), ('c3B', (99, 68, 156) * 4,))

        self.agent8 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [self.p8[0] - 30 / 2, self.p8[1] - 30 / 2,  # location
                     self.p8[0] - 30 / 2, self.p8[1] + 30 / 2,
                     self.p8[0] + 30 / 2, self.p8[1] + 30 / 2,
                     self.p8[0] + 30 / 2, self.p8[1] - 30 / 2]), ('c3B', (245, 168, 156) * 4,))
        self.agent7 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [self.p7[0] - 30 / 2, self.p7[1] - 30 / 2,  # location
                     self.p7[0] - 30 / 2, self.p7[1] + 30 / 2,
                     self.p7[0] + 30 / 2, self.p7[1] + 30 / 2,
                     self.p7[0] + 30 / 2, self.p7[1] - 30 / 2]), ('c3B', (15, 65, 156) * 4,))




    def on_draw(self):
        self.clear()  # 清屏
        self.batch.draw()  # 画上 batch 里面的内容

    #更新agent的位置
    def _update_arm(self,p1,p2,p3,p4,p5,p6,p7,p8):
        px=p1[0]
        py= p1[1]
        x=p2[0]
        y=p2[1]
        px1=p3[0]
        py1=p3[1]
        x1=p4[0]
        y1=p4[1]
        x2 = p5[0]
        y2 = p5[1]
        x3 = p6[0]
        y3 = p6[1]
        x4 = p7[0]
        y4 = p7[1]
        x5 = p8[0]
        y5 = p8[1]
       # (px1, py1) = self.p3
       # (x1, y1) = self.p4
        a=(px-15,py-15)
        b=(px-15,py+15)
        c=(px+15,py+15)
        d=(px+15,py-15)
        # 将点信息都放入手臂显示中
        aa = (x - 15, y - 15)
        bb = (x - 15, y + 15)
        cc = (x + 15, y + 15)
        dd = (x + 15, y - 15)

        a1 = (px1 - 15, py1 - 15)
        b1 = (px1 - 15, py1 + 15)
        c1 = (px1 + 15, py1 + 15)
        d1 = (px1 + 15, py1 - 15)

        a2 = (x1 - 15, y1 - 15)
        b2 = (x1 - 15, y1 + 15)
        c2 = (x1 + 15, y1 + 15)
        d2 = (x1 + 15, y1 - 15)


        q1= (x2-15,y2-15)
        w1= (x2-15,y2+15)
        e1 = (x2 + 15, y2 + 15)
        r1 = (x2 + 15, y2 - 15)

        q2 = (x3 - 15, y3 - 15)
        w2 = (x3 - 15, y3 + 15)
        e2 = (x3 + 15, y3 + 15)
        r2 = (x3 + 15, y3 - 15)

        q3 = (x4 - 15, y4 - 15)
        w3 = (x4 - 15, y4 + 15)
        e3 = (x4 + 15, y4 + 15)
        r3 = (x4 + 15, y4 - 15)

        q4 = (x5 - 15, y5 - 15)
        w4 = (x5 - 15, y5 + 15)
        e4 = (x5 + 15, y5 + 15)
        r4 = (x5 + 15, y5 - 15)

        self.agent1.vertices = np.concatenate((a,b,c,d))
        self.agent2.vertices = np.concatenate((aa, bb, cc, dd))
        self.agent3.vertices = np.concatenate((a1, b1, c1, d1))
        self.agent4.vertices = np.concatenate((a2, b2, c2, d2))
        self.agent5.vertices = np.concatenate((q1, w1, e1, r1))
        self.agent6.vertices = np.concatenate((q2, w2, e2, r2))
        self.agent7.vertices = np.concatenate((q3, w3, e3, r3))
        self.agent8.vertices = np.concatenate((q4, w4, e4, r4))
    def render(self,p1,p2,p3,p4,p5,p6,p7,p8):
        self._update_arm(p1,p2,p3,p4,p5,p6,p7,p8)  # 更新手臂内容 (暂时没有变化)
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

class  ArmEnv(object):
    viewer = None  # 首先没有 viewer
    dt =5# refresh rate
    #把速度大小限制在7之内
    action_bound = [-7, 7]
    goal1 = {'x': 400., 'y': 600., 'l': 50}
    agent1 = {'x': 400.,'y': 200., 'l': 10}


    goal2 = {'x': 400., 'y': 200., 'l': 50}
    agent2 = {'x': 400., 'y': 600., 'l': 10}

    goal3 = {'x': 600., 'y': 400., 'l': 50}
    agent3 = {'x': 200., 'y': 400., 'l': 10}

    goal4 = {'x': 200., 'y': 400., 'l': 50}
    agent4 = {'x': 600., 'y': 400., 'l': 10}

    ws_model = dict()
    # robot radius
    ws_model['robot_radius'] = 30

    ws_model['circular_obstacles'] = []

    ws_model['boundary'] = []


    def __init__(self,n):
        #初始化Agent
        self.n=n
        self.agent1 = Agent([200 ,100], [0 ,0], [500,400],10,50)
        self.agent2 = Agent([300, 100], [0, 0], [400,400],10,50)
        self.agent3 = Agent([400, 100], [0, 0], [300, 400], 10, 50)
        self.agent4 = Agent([500, 100], [0, 0], [200, 400], 10, 50)
        self.agent5 = Agent([500, 400], [0, 0], [200, 100], 10, 50)
        self.agent6 = Agent([400, 400], [0, 0], [300, 100], 10, 50)
        self.agent7 = Agent([300, 400], [0, 0], [400, 100], 10, 50)
        self.agent8 = Agent([200, 400], [0, 0], [500, 100], 10, 50)
    def computerdis(self,x,y):
        return math.sqrt(np.square(x[0]-y[0])+np.square(x[1]-y[1]))
    def step(self,action):
        #action是列表里面套array数组
        #reward直接在step中定义即可
        Agent = []
        done=[False,False,False,False,False,False,False,False]
        a1=action[0]
        a2=action[1]
        a3=action[2]
        a4=action[3]
        a5 = action[4]
        a6 = action[5]
        a7 = action[6]
        a8 = action[7]
        reward=[]
        r1=0
        r2=0
        r3=0
        r4=0
        r5=0
        r6=0
        r7=0
        r8=0
        obs=[]
        #1.首先计算reward,先计算出RVO速度和当前速度即action（a1,a2）做对比
        P=np.zeros((self.n,2))
        P[0]=copy.deepcopy(self.agent1.p)
        P[1]=copy.deepcopy(self.agent2.p)
        P[2]=copy.deepcopy(self.agent3.p)
        P[3] = copy.deepcopy(self.agent4.p)
        P[4] = copy.deepcopy(self.agent5.p)
        P[5] = copy.deepcopy(self.agent6.p)
        P[6] = copy.deepcopy(self.agent7.p)
        P[7] = copy.deepcopy(self.agent8.p)
        V = np.zeros((self.n,2))
        V[0] = copy.deepcopy(self.agent1.v)
        V[1] = copy.deepcopy(self.agent2.v)
        V[2] = copy.deepcopy(self.agent3.v)
        V[3] = copy.deepcopy(self.agent4.v)
        V[4] = copy.deepcopy(self.agent5.v)
        V[5] = copy.deepcopy(self.agent6.v)
        V[6] = copy.deepcopy(self.agent7.v)
        V[7] = copy.deepcopy(self.agent8.v)
        G = np.zeros((self.n,2))
        G[0] = copy.deepcopy(self.agent1.g)
        G[1] = copy.deepcopy(self.agent2.g)
        G[2] = copy.deepcopy(self.agent3.g)
        G[3] = copy.deepcopy(self.agent4.g)
        G[4] = copy.deepcopy(self.agent5.g)
        G[5] = copy.deepcopy(self.agent6.g)
        G[6] = copy.deepcopy(self.agent7.g)
        G[7] = copy.deepcopy(self.agent8.g)

        V_max = [1.0 for i in range(len(P))]
        # 计算首选速度
        V_des = compute_V_des(P,G,V_max)
        # 计算RVO速度
        V = RVO_update(P, V_des, V, self.ws_model)

        dists1 = np.sqrt(np.sum(np.square(a1 - V[0])))
        dists2 = np.sqrt(np.sum(np.square(a2 - V[1])))
        dists3 = np.sqrt(np.sum(np.square(a3 - V[2])))
        dists4 = np.sqrt(np.sum(np.square(a4 - V[3])))
        dists5 = np.sqrt(np.sum(np.square(a5 - V[4])))
        dists6 = np.sqrt(np.sum(np.square(a6 - V[5])))
        dists7 = np.sqrt(np.sum(np.square(a7 - V[6])))
        dists8 = np.sqrt(np.sum(np.square(a8 - V[7])))
        r1 += math.exp(-dists1)*0.3
        r2 += math.exp(-dists2)*0.3
        r3 += math.exp(-dists3) * 0.3
        r4 += math.exp(-dists4) * 0.3
        r5 += math.exp(-dists5) * 0.3
        r6 += math.exp(-dists6) * 0.3
        r7 += math.exp(-dists7) * 0.3
        r8 += math.exp(-dists8) * 0.3
        #更新智能体状态,获得新obs

        self.agent1.p[0] += a1[0] * self.dt
        self.agent1.p[1] += a1[1] * self.dt
        self.agent2.p[0] += a2[0] * self.dt
        self.agent2.p[1] += a2[1] * self.dt
        self.agent3.p[0] += a3[0] * self.dt
        self.agent3.p[1] += a3[1] * self.dt
        self.agent4.p[0] += a4[0] * self.dt
        self.agent4.p[1] += a4[1] * self.dt
        self.agent5.p[0] += a5[0] * self.dt
        self.agent5.p[1] += a5[1] * self.dt
        self.agent6.p[0] += a6[0] * self.dt
        self.agent6.p[1] += a6[1] * self.dt
        self.agent7.p[0] += a7[0] * self.dt
        self.agent7.p[1] += a7[1] * self.dt
        self.agent8.p[0] += a8[0] * self.dt
        self.agent8.p[1] += a8[1] * self.dt
        self.agent1.v=a1
        self.agent2.v=a2
        self.agent3.v = a3
        self.agent4.v = a4
        self.agent5.v = a5
        self.agent6.v = a6
        self.agent7.v = a7
        self.agent8.v = a8
        d1 = -self.computerdis(self.agent1.p,self.agent1.g)*0.001
        d2 = -self.computerdis(self.agent2.p, self.agent2.g) * 0.001
        d3 = -self.computerdis(self.agent3.p, self.agent3.g) * 0.001
        d4 = -self.computerdis(self.agent4.p, self.agent4.g) * 0.001
        d5 = -self.computerdis(self.agent5.p, self.agent5.g) * 0.001
        d6 = -self.computerdis(self.agent6.p, self.agent6.g) * 0.001
        d7 = -self.computerdis(self.agent7.p, self.agent7.g) * 0.001
        d8 = -self.computerdis(self.agent8.p, self.agent8.g) * 0.001
        r1+=d1*0.7
        r2+=d2*0.7
        r3 += d3 * 0.7
        r4 += d4 * 0.7
        r5 += d5 * 0.7
        r6 += d6 * 0.7
        r7 += d7 * 0.7
        r8 += d8 * 0.7

        Agent.append(self.agent1)
        Agent.append(self.agent2)
        Agent.append(self.agent3)
        Agent.append(self.agent4)
        Agent.append(self.agent5)
        Agent.append(self.agent6)
        Agent.append(self.agent7)
        Agent.append(self.agent8)
        x = combinations(Agent, 2)

        aobs= np.array([self.agent1.p[0],self.agent1.p[1],a1[0],a1[1],self.agent1.g[0],self.agent1.g[1],self.agent1.r,self.agent1.gr])*0.001
        bobs = np.array([self.agent2.p[0], self.agent2.p[1], a2[0], a2[1], self.agent2.g[0], self.agent2.g[1],
                        self.agent2.r, self.agent2.gr])*0.001
        cobs = np.array(
            [self.agent3.p[0], self.agent3.p[1], a3[0], a3[1], self.agent3.g[0], self.agent3.g[1], self.agent3.r,
             self.agent3.gr]) * 0.001
        dobs = np.array([self.agent4.p[0], self.agent4.p[1], a4[0], a4[1], self.agent4.g[0], self.agent4.g[1],
                         self.agent4.r, self.agent4.gr]) * 0.001
        eobs = np.array(
            [self.agent5.p[0], self.agent5.p[1], a5[0], a5[1], self.agent5.g[0], self.agent5.g[1], self.agent5.r,
             self.agent5.gr]) * 0.001
        fobs = np.array([self.agent6.p[0], self.agent6.p[1], a6[0], a6[1], self.agent6.g[0], self.agent6.g[1],
                         self.agent6.r, self.agent6.gr]) * 0.001
        gobs = np.array(
            [self.agent7.p[0], self.agent7.p[1], a7[0], a7[1], self.agent7.g[0], self.agent7.g[1], self.agent7.r,
             self.agent7.gr]) * 0.001
        hobs = np.array([self.agent8.p[0], self.agent8.p[1], a8[0], a8[1], self.agent8.g[0], self.agent8.g[1],
                         self.agent8.r, self.agent8.gr]) * 0.001
        obs.append(aobs)
        obs.append(bobs)
        obs.append(cobs)
        obs.append(dobs)
        obs.append(eobs)
        obs.append(fobs)
        obs.append(gobs)
        obs.append(hobs)
        if self.collection(self.agent1.p[0],self.agent1.p[1],30,self.agent1.g[0],self.agent1.g[1],50):
            r1+=1
            done[0]=True
        if self.collection(self.agent2.p[0],self.agent2.p[1],30,self.agent2.g[0],self.agent2.g[1],50):
            r2+=1
            done[1]=True
        if self.collection(self.agent3.p[0],self.agent3.p[1],30,self.agent3.g[0],self.agent3.g[1],50):
            r3+=1
            done[2]=True
        if self.collection(self.agent4.p[0],self.agent4.p[1],30,self.agent4.g[0],self.agent4.g[1],50):
            r4+=1
            done[3]=True
        if self.collection(self.agent5.p[0],self.agent5.p[1],30,self.agent5.g[0],self.agent5.g[1],50):
            r5+=1
            done[4]=True
        if self.collection(self.agent6.p[0],self.agent6.p[1],30,self.agent6.g[0],self.agent6.g[1],50):
            r6+=1
            done[5]=True
        if self.collection(self.agent7.p[0],self.agent7.p[1],30,self.agent7.g[0],self.agent7.g[1],50):
            r7+=1
            done[6]=True
        if self.collection(self.agent8.p[0],self.agent8.p[1],30,self.agent8.g[0],self.agent8.g[1],50):
            r8+=1
            done[7]=True
        if self.collection(self.agent1.p[0], self.agent1.p[1], 30, self.agent2.p[0], self.agent2.p[1], 30):
            r1-=1
            r2-=1
        for test in x:
            if self.collection(test[0].p[0], test[0].p[1], 30,test[1].p[0], test[1].p[1], 30):
                if test[0]==self.agent1:
                    r1-=10
                if test[0] == self.agent2:
                    r2 -= 10
                if test[0]==self.agent3:
                    r3-=10
                if test[0] == self.agent4:
                    r4 -= 10
                if test[0]==self.agent5:
                    r5-=10
                if test[0] == self.agent6:
                    r6 -= 10
                if test[0]==self.agent7:
                    r7-=10
                if test[0] == self.agent8:
                    r8 -= 10
                if test[1]==self.agent1:
                    r1-=10
                if test[1] == self.agent2:
                    r2 -= 10
                if test[1]==self.agent3:
                    r3-=10
                if test[1] == self.agent4:
                    r4 -= 10
                if test[1]==self.agent5:
                    r5-=10
                if test[1] == self.agent6:
                    r6 -= 10
                if test[1]==self.agent7:
                    r7-=10
                if test[1] == self.agent8:
                    r8 -= 10


        reward.append(r1)
        reward.append(r2)
        reward.append(r3)
        reward.append(r4)
        reward.append(r5)
        reward.append(r6)
        reward.append(r7)
        reward.append(r8)
        return obs,reward,done


#重新写reset环境，使得智能体出现在不同位置，并且不会与障碍物重叠
    def reset(self):
        obs=[]
        self.agent1.setatt([200,100],  [0,0],  [500,400],10,50)
        self.agent2.setatt([300,100],  [0, 0], [400,400], 10, 50)
        self.agent3.setatt([400, 100], [0, 0], [300, 400], 10, 50)
        self.agent4.setatt([500, 100], [0, 0], [200, 400], 10, 50)
        self.agent5.setatt([500, 400], [0, 0], [200, 100], 10, 50)
        self.agent6.setatt([400, 400], [0, 0], [300, 100], 10, 50)
        self.agent7.setatt([300, 400], [0, 0], [400, 100], 10, 50)
        self.agent8.setatt([200, 400], [0, 0], [500, 100], 10, 50)
        aobs = np.array(
            [self.agent1.p[0], self.agent1.p[1], self.agent1.v[0], self.agent1.v[1], self.agent1.g[0], self.agent1.g[1], self.agent1.r,
             self.agent1.gr]) * 0.001
        bobs = np.array([self.agent2.p[0], self.agent2.p[1], self.agent2.v[0], self.agent2.v[1], self.agent2.g[0], self.agent2.g[1],
                         self.agent2.r, self.agent2.gr]) * 0.001
        cobs = np.array(
            [self.agent3.p[0], self.agent3.p[1], self.agent3.v[0], self.agent3.v[1], self.agent3.g[0], self.agent3.g[1], self.agent3.r,
             self.agent3.gr]) * 0.001
        dobs = np.array([self.agent4.p[0], self.agent4.p[1], self.agent4.v[0], self.agent1.v[1], self.agent4.g[0], self.agent4.g[1],
                         self.agent4.r, self.agent4.gr]) * 0.001
        eobs = np.array(
            [self.agent5.p[0], self.agent5.p[1], self.agent1.v[0], self.agent1.v[1], self.agent5.g[0], self.agent5.g[1], self.agent5.r,
             self.agent5.gr]) * 0.001
        fobs = np.array([self.agent6.p[0], self.agent6.p[1], self.agent1.v[0], self.agent1.v[1], self.agent6.g[0], self.agent6.g[1],
                         self.agent6.r, self.agent6.gr]) * 0.001
        gobs = np.array(
            [self.agent7.p[0], self.agent7.p[1], self.agent1.v[0], self.agent1.v[1], self.agent7.g[0], self.agent7.g[1], self.agent7.r,
             self.agent7.gr]) * 0.001
        hobs = np.array([self.agent8.p[0], self.agent8.p[1], self.agent1.v[0], self.agent1.v[1], self.agent8.g[0], self.agent8.g[1],
                         self.agent8.r, self.agent8.gr]) * 0.001
        obs.append(aobs)
        obs.append(bobs)
        obs.append(cobs)
        obs.append(dobs)
        obs.append(eobs)
        obs.append(fobs)
        obs.append(gobs)
        obs.append(hobs)
        return obs


    def test_RVO(self):
        P = np.zeros((self.n, 2))
        P[0] = copy.deepcopy(self.agent1.p)
        P[1] = copy.deepcopy(self.agent2.p)

        V = np.zeros((self.n, 2))
        V[0] = copy.deepcopy(self.agent1.v)
        V[1] = copy.deepcopy(self.agent2.v)

        G = np.zeros((self.n, 2))
        G[0] = copy.deepcopy(self.agent1.g)
        G[1] = copy.deepcopy(self.agent2.g)

        V_max = [1.0 for i in range(len(P))]
        # 计算首选速度
        V_des = compute_V_des(P, G, V_max)
        # 计算RVO速度
        V = RVO_update(P, V_des, V, self.ws_model)
        self.agent1.p[0] += V[0][0] * self.dt
        self.agent1.p[1] += V[0][1] * self.dt
        self.agent2.p[0] += V[1][0] * self.dt
        self.agent2.p[1] += V[1][1] * self.dt
        print(V[0])
    def render(self):
        if self.viewer is None:  # 如果调用了 render, 而且没有 viewer, 就生成一个
            self.viewer = Viewer(self.agent1,self.agent2,self.agent3,self.agent4,self.agent5,self.agent6,self.agent7,self.agent8)
        self.viewer.render(self.agent1.p,self.agent2.p,self.agent3.p,self.agent4.p,self.agent5.p,self.agent6.p,self.agent7.p,self.agent8.p)  # 使用 Viewer 中的 render 功能

    def sample_action(self):
        return np.random.rand(2)*20-10  # two radians

    def collection(self,x1,y1,h1,x2,y2,h2):
        #如果碰撞，则返回真
        if abs(x1-x2)< h1/2+h2/2 and abs(y1-y2)<h1/2+h2/2:
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
    a = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
    print(a)
    print(a[:,:-1])


