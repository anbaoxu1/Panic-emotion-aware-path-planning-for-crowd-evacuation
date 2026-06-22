import xlrd
import numpy as np
import random
import scipy.signal as ss
import matplotlib.pyplot as plt
data=xlrd.open_workbook("./test2.xlsx")
sheet=data.sheets()[0]
n_of_rows=sheet.nrows
list=[]
for i in range(n_of_rows):
    if i>=3:
        list.append(sheet.row_values(i))
print(list)
x=[]
y=[]
v=[]
xx=[]
yy=[]
for i in range(len(list)):
    x.append(list[i][1])
    y.append(list[i][2])
    v.append(list[i][3]*0.01)
xx = np.loadtxt('l1.txt', dtype=np.float32)
yy = np.loadtxt('l2.txt', dtype=np.float32)
"""
for i in range(len(list)):
    xx.append(list[i][1])
    te=random.random()
    if te>0.4:
        yy.append(list[i][2]+random.uniform(-20,20))
    else:
        yy.append(list[i][2])
"""
"""      
np.savetxt('x.txt', xx, fmt='%0.2f')
np.savetxt('y.txt', yy, fmt='%0.2f')
"""
    #v.append(list[i][3]*0.01)
#yy = np.loadtxt('y.txt', dtype=np.float32)

tmp_smooth1 = ss.savgol_filter(yy,51,11)
plt.figure()  # 整个现实图（框架）的大小
plt.plot(x, y,  linewidth=2)
plt.plot(xx,tmp_smooth1,'r--',  linewidth=2)
#plt.plot(x[0:200], yy[0:200],  linewidth=2)
plt.xlabel('X')
plt.legend(('real', 'simulation',))
plt.ylabel("Y")
plt.title("Comparison of real path and simulation path without data")
plt.show()

