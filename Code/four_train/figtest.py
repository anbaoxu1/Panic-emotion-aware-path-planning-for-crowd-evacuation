import numpy as np
import matplotlib.pyplot as plt
import copy
x=np.linspace(0,10,100)
y=np.sin(x)
y1=np.cos(x)
plt.figure(figsize=(5,5)) #整个现实图（框架）的大小
plt.plot(x,y,label="$sin(x)$",linewidth=1)
plt.plot(x,y1,label="$(x)$",linewidth=1)
plt.xlabel('Time(s)')
plt.legend(('sinx', 'cosx', ))
plt.ylabel("Volt")
plt.title("Python chart")
plt.show()
if __name__ == '__main__':
    x = np.loadtxt('eps.txt', dtype=np.float32)

    y = np.loadtxt('e1.txt', dtype=np.float32)
    yy = np.loadtxt('e2.txt', dtype=np.float32)
    plt.figure()  # 整个现实图（框架）的大小
    plt.plot(x[0:200], y[0:200],  linewidth=2)
    plt.plot(x[0:200], yy[0:200],  linewidth=2)
    plt.xlabel('episode')
    plt.legend(('using RVO', 'no RVO',))
    plt.ylabel("Reward")

    plt.show()
