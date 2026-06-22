import pyglet
import numpy as np
import copy
from RVO import RVO_update, reach, compute_V_des, reach
import time
#reset,step,reward
class Viewer(pyglet.window.Window):
    def __init__(self, agent_info1, agent_info2,agent_info3,agent_info4):
        ...
        # 添加 arm 信息

        self.p1=agent_info1['p']
        self.g1=agent_info1['g']
        self.p2 = agent_info2['p']
        self.g2 = agent_info2['g']
        self.p3 = agent_info3['p']
        self.g3 = agent_info3['g']
        self.p4 = agent_info4['p']
        self.g4 = agent_info4['g']
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
            ('v2f', [self.g1[0] - 50 / 2, self.g1[1] - 50 / 2,
                     self.g1[0] - 50 / 2, self.g1[1] + 50  / 2,
                     self.g1[0] + 50  / 2, self.g1[1] + 50 / 2,
                     self.g1[0] + 50  / 2, self.g1[1] - 50  / 2]),
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
        self.agent3= self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [self.p4[0] - 30 / 2, self.p4[1] - 30 / 2,  # location
                     self.p4[0] - 30 / 2, self.p4[1] + 30 / 2,
                     self.p4[0] + 30 / 2, self.p4[1] + 30 / 2,
                     self.p4[0] + 30 / 2, self.p4[1] - 30 / 2]), ('c3B', (15, 65, 156) * 4,))
        self.agent4 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [self.p2[0] - 30 / 2, self.p2[1] - 30 / 2,  # location
                     self.p2[0] - 30 / 2, self.p2[1] + 30 / 2,
                     self.p2[0] + 30 / 2, self.p2[1] + 30 / 2,
                     self.p2[0] + 30 / 2, self.p2[1] - 30 / 2]), ('c3B', (245, 168, 156) * 4,))


    def render(self):
        self._update_arm()  # 更新手臂内容 (暂时没有变化)
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()  # 清屏
        self.batch.draw()  # 画上 batch 里面的内容

    #更新agent的位置
    def _update_arm(self):
        (px, py) = self.p1
        (x, y) = self.p2
        (px1, py1) = self.p3
        (x1, y1) = self.p4
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
        self.agent1.vertices = np.concatenate((a,b,c,d))
        self.agent2.vertices = np.concatenate((aa, bb, cc, dd))
        self.agent3.vertices = np.concatenate((a1, b1, c1, d1))
        self.agent4.vertices = np.concatenate((a2, b2, c2, d2))

class  ArmEnv(object):
    viewer = None  # 首先没有 viewer
    dt =3# refresh rate
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
    ws_model['robot_radius'] = 50

    ws_model['circular_obstacles'] = []

    ws_model['boundary'] = []


    def __init__(self):
        self.agent_info1 = np.zeros(
            2, dtype=[('p', np.float32), ('v', np.float32),('g', np.float32)])
        self.agent_info1['p'] = (self.agent1['x'],self.agent1['y'])
        self.agent_info1['v'] = (0,0)
        self.agent_info1['g'] = (self.goal1['x'],self.goal1['y'])

        self.agent_info2= np.zeros(
            2, dtype=[('p', np.float32), ('v', np.float32), ('g', np.float32)])
        self.agent_info2['p'] = (self.agent2['x'], self.agent2['y'])
        self.agent_info2['v'] = (0, 0)
        self.agent_info2['g'] = (self.goal2['x'], self.goal2['y'])

        self.agent_info3= np.zeros(
            2, dtype=[('p', np.float32), ('v', np.float32), ('g', np.float32)])
        self.agent_info3['p'] = (self.agent3['x'], self.agent3['y'])
        self.agent_info3['v'] = (0, 0)
        self.agent_info3['g'] = (self.goal3['x'], self.goal3['y'])

        self.agent_info4 = np.zeros(
            2, dtype=[('p', np.float32), ('v', np.float32), ('g', np.float32)])
        self.agent_info4['p'] = (self.agent4['x'], self.agent4['y'])
        self.agent_info4['v'] = (0, 0)
        self.agent_info4['g'] = (self.goal4['x'], self.goal4['y'])

    def step(self):

        P=np.zeros((4,2))
        P[0]=copy.deepcopy(self.agent_info1['p'])
        P[1]=copy.deepcopy(self.agent_info2['p'])
        P[2] = copy.deepcopy(self.agent_info3['p'])
        P[3] = copy.deepcopy(self.agent_info4['p'])
        V = np.zeros((4,2))
        V[0] = copy.deepcopy(self.agent_info1['v'])
        V[1] = copy.deepcopy(self.agent_info2['v'])
        V[2] = copy.deepcopy(self.agent_info3['v'])
        V[3] = copy.deepcopy(self.agent_info4['v'])
        G = np.zeros((4,2))
        G[0] = copy.deepcopy(self.agent_info1['g'])
        G[1] = copy.deepcopy(self.agent_info2['g'])
        G[2] = copy.deepcopy(self.agent_info3['g'])
        G[3] = copy.deepcopy(self.agent_info4['g'])

        V_max = [1.0 for i in range(len(P))]
        # 计算首选速度
        V_des = compute_V_des(P,G,V_max)
        # 计算RVO速度
        V = RVO_update(P, V_des, V, self.ws_model)
        print(V[0])
        for i in range(len(P)):
            P[i][0] += V[i][0] * self.dt
            P[i][1] += V[i][1] * self.dt
        self.agent_info1['p']=  copy.deepcopy(P[0])
        self.agent_info2['p'] = copy.deepcopy(P[1])
        self.agent_info1['v'] = copy.deepcopy(V[0])
        self.agent_info2['v'] = copy.deepcopy(V[1])
        self.agent_info3['p'] = copy.deepcopy(P[2])
        self.agent_info4['p'] = copy.deepcopy(P[3])
        self.agent_info3['v'] = copy.deepcopy(V[2])
        self.agent_info4['v'] = copy.deepcopy(V[3])

        return self.agent_info1,self.agent_info2,self.agent_info3,self.agent_info4


#重新写reset环境，使得智能体出现在不同位置，并且不会与障碍物重叠
    def reset(self):
        time.sleep(0.1)
        self.agent_info1['p'] = (self.agent1['x'], self.agent1['y'])
        self.agent_info1['v'] = (0, 0)
        self.agent_info1['g'] = (self.goal1['x'], self.goal1['y'])

        self.agent_info2['p'] = (self.agent2['x'], self.agent2['y'])
        self.agent_info2['v'] = (0, 0)
        self.agent_info2['g'] = (self.goal2['x'], self.goal2['y'])

        self.agent_info3['p'] = (self.agent3['x'], self.agent3['y'])
        self.agent_info3['v'] = (0, 0)
        self.agent_info3['g'] = (self.goal3['x'], self.goal3['y'])

        self.agent_info4['p'] = (self.agent4['x'], self.agent4['y'])
        self.agent_info4['v'] = (0, 0)
        self.agent_info4['g'] = (self.goal4['x'], self.goal4['y'])

        return self.agent_info1,self.agent_info2

    def render(self):
        if self.viewer is None:  # 如果调用了 render, 而且没有 viewer, 就生成一个
            self.viewer = Viewer(self.agent_info1,self.agent_info2,self.agent_info3,self.agent_info4)
        self.viewer.render()  # 使用 Viewer 中的 render 功能

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



if __name__ == '__main__':
    env = ArmEnv()
    while True:
        env.render()
        env.step()
        if env.agent_info1['p'][1]>600:
            env.reset()

        #env.reset()
