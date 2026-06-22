import numpy as np
import matplotlib.pyplot as plt
import copy

if __name__ == '__main__':
    x = np.array([1000,2000,3000,4000,5000])

    y = np.array([0.75,2.11,2.7,3.8,4.5])
    yy = np.array([0.42,0.67,1.09,1.5,1.9])
    plt.figure()  # 整个现实图（框架）的大小
    plt.plot(x[0:200], y[0:200], 'r-o', linewidth=2)
    plt.plot(x[0:200], yy[0:200], 'b-o', linewidth=2)
    plt.xlabel('agent number')
    plt.legend(('RVO', 'Our method',))
    plt.ylabel("Time(s)")

    plt.show()