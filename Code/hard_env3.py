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
单智能体环境
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
class Obstacle():
    def __init__(self,x,y,width,heigh):
        self.x=x
        self.y=y
        #此处宽高为实际宽高的一半
        self.w=width
        self.h=heigh





class Viewer(pyglet.window.Window):
    #传一个agent数组即可
    def __init__(self,agentinfo,ox):
        ...
        # 初始化位置数组和目标数组
        self.p=[]
        self.g=[]
        self.o=[]
        #画图数组
        self.pos=[]
        self.goal=[]
        self.obs=[]
        for info in agentinfo:
            self.p.append(info.p)
            self.g.append(info.g)
        for ob in ox:
            self.o.append(ob)

        # 创建窗口的继承
        # vsync 如果是 True, 按屏幕频率刷新, 反之不按那个频率
        super(Viewer, self).__init__(width=800, height=800, resizable=False, caption='Agent', vsync=True)
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

        for index in range(len(self.o)):
            self.obs.append(self.batch.add(
                4, pyglet.gl.GL_QUADS, None,  # 4 corners
                ('v2f', [self.o[index].x - self.o[index].w, self.o[index].y - self.o[index].h,
                         self.o[index].x - self.o[index].w, self.o[index].y + self.o[index].h,
                         self.o[index].x + self.o[index].w, self.o[index].y + self.o[index].h,
                         self.o[index].x + self.o[index].w, self.o[index].y - self.o[index].h]),
                ('c3B', (0,0,255) * 4), ))

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

    def __init__(self,n):
        #初始化Agent
        self.n=n
        self.Agent=[]
        self.Obstacle=[]
        for i in range(n):
            x = np.random.randint(50,750,2)
            y = np.array([780,350])
           # y=np.array([500,500])
            #y目标应为固定值
            agent = Agent(x, [0, 0], y)
            self.Agent.append(agent)
        o1 = Obstacle(35,400,5,380)
        o2 = Obstacle(270, 775, 230, 5)
        o3 = Obstacle(720, 775, 70, 5)
        o4 = Obstacle(785, 585, 5, 170)
        o5 = Obstacle(785, 160, 5, 140)
        o6 = Obstacle(410, 25, 370, 5)
        o7 = Obstacle(220, 405, 180, 5)
        o8 = Obstacle(405, 400, 5, 200)
        o9 = Obstacle(405, 65, 5, 35)
        self.Obstacle.append(o1)
        self.Obstacle.append(o2)
        self.Obstacle.append(o3)
        self.Obstacle.append(o4)
        self.Obstacle.append(o5)
        self.Obstacle.append(o6)
        self.Obstacle.append(o7)
        self.Obstacle.append(o8)
        self.Obstacle.append(o9)
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
        fla=False
        done=[]
        #初始化done元组
        for index in range(len(self.Agent)):
            done.append(False)
        reward=[]
        obs=[]
        #1.首先reward包含距离，与障碍物（墙壁）碰撞，会减一，
        pre_pos=np.zeros((self.n,2))
        #更新智能体状态
        for index in range(len(action)):
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
            reward.append(asd)

        """
        计算离当前智能体最近的3个智能体的状态（距离目标位置，相对距离，相对速度）
        """
        for index in range(len(self.Agent)):
            relate_p = []
            relate_g = []

            pp = []
            gg = []

            relate_g.append(self.Agent[index].p - self.Agent[index].g)
            for i in range(len(self.Obstacle)):
                    x=self.Obstacle[i].x - self.Agent[index].p[0]
                    y=self.Obstacle[i].y - self.Agent[index].p[1]
                    relate_p.append(np.array([x,y]))
            for b in relate_p:
                pp.append(b[0])
                pp.append(b[1])
            for g in relate_g:
                gg.append(g[0])
                gg.append(g[1])
            xobs = np.concatenate([np.array(
                [self.Agent[index].p[0], self.Agent[index].p[1], self.Agent[index].v[0], self.Agent[index].v[1],
                 self.Agent[index].g[0], self.Agent[index].g[1],
                 ])] + [pp] + [gg]) * 0.001

            obs.append(xobs)

        for index in range(len(self.Agent)):
            if self.dist1(self.Agent[index]):
                reward[index] += 1
                done[index] = True

        for index in range(len(self.Agent)):
            for i in range(len(self.Obstacle)):
                if self.collection(self.Agent[index].p[0],self.Agent[index].p[1],40,40,
                                   self.Obstacle[i].x,self.Obstacle[i].y,self.Obstacle[i].w*2,self.Obstacle[i].h*2):
                    reward[index]-=5
                    fla=True
        return obs,reward,done,fla

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
            relate_p = []
            relate_g = []
            pp = []
            gg = []
            relate_g.append(self.Agent[index].p - self.Agent[index].g)
            for i in range(len(self.Obstacle)):
                x = self.Obstacle[i].x - self.Agent[index].p[0]
                y = self.Obstacle[i].y - self.Agent[index].p[1]
                relate_p.append(np.array([x, y]))
            for b in relate_p:
                pp.append(b[0])
                pp.append(b[1])
            for g in relate_g:
                gg.append(g[0])
                gg.append(g[1])
            xobs = np.concatenate([np.array(
                [self.Agent[index].p[0], self.Agent[index].p[1], self.Agent[index].v[0], self.Agent[index].v[1],
                 self.Agent[index].g[0], self.Agent[index].g[1],
                 ])] + [pp] + [gg]) * 0.001
            kk.append(xobs)
        return len(kk[0])

#重新写reset环境，使得智能体出现在不同位置，并且不会与障碍物重叠
    def reset(self):
        obs=[]
        for index in range(len(self.Agent)):
            x = np.random.randint(0, 800, 2)
            a = np.random.randint(0, 2, 1)
            if a==0:
                y = np.array([780,360])
            else:
                y = np.array([575,800])
            self.Agent[index].setatt(x,np.array([0,0]),y)
        """
        self.Agent[0].setatt(np.array([400,200]), [0, 0], np.array([400,600]))
        self.Agent[1].setatt(np.array([400, 600]), [0, 0], np.array([400, 200]))
        self.Agent[2].setatt(np.array([200, 400]), [0, 0], np.array([600, 400]))
        self.Agent[3].setatt(np.array([600, 400]), [0, 0], np.array([200, 400]))
        self.Agent[4].setatt(np.array([600, 200]), [0, 0], np.array([200, 600]))
        """
        for index in range(len(self.Agent)):
            relate_p = []
            relate_g = []
            pp = []
            gg = []
            relate_g.append(self.Agent[index].p - self.Agent[index].g)
            for i in range(len(self.Obstacle)):
                x = self.Obstacle[i].x - self.Agent[index].p[0]
                y = self.Obstacle[i].y - self.Agent[index].p[1]
                relate_p.append(np.array([x, y]))
            for b in relate_p:
                pp.append(b[0])
                pp.append(b[1])
            for g in relate_g:
                gg.append(g[0])
                gg.append(g[1])
            xobs = np.concatenate([np.array(
                [self.Agent[index].p[0], self.Agent[index].p[1], self.Agent[index].v[0], self.Agent[index].v[1],
                 self.Agent[index].g[0], self.Agent[index].g[1],
                 ])] + [pp] + [gg]) * 0.001
            obs.append(xobs)
        return obs



    def render(self):
        if self.viewer is None:  # 如果调用了 render, 而且没有 viewer, 就生成一个
            self.viewer = Viewer(self.Agent,self.Obstacle)
        p=[]
        for index in range(len(self.Agent)):
            p.append(self.Agent[index].p)
        g = []
        for index in range(len(self.Agent)):
            g.append(self.Agent[index].g)
        self.viewer.render(p,g)  # 使用 Viewer 中的 render 功能

    def sample_action(self):
        return np.random.rand(2)*20-10  # two radians

    def collection(self,x1,y1,w1,h1,x2,y2,w2,h2):
        #如果碰撞，则返回真
        if abs(x1-x2)< w1/2+w2/2 and abs(y1-y2)<h1/2+h2/2:
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

    def round2(value, num=2):
        tmp = str(value) + '1'
        return round(float(tmp), num)
if __name__ == '__main__':
    x1=np.array([1,2,3])
    x2= np.array([1, 2, 3,4])
    x=np.hstack((x1,x2))
    print(x)


