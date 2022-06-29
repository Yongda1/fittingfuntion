import numpy as np
import scipy
from scipy.linalg import lstsq
from scipy.optimize import fmin
np.set_printoptions(precision=16)
from pylab import plot, show
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#--------------------------------------------------
'''
This script works for 7-points fitting.
'''
#Before use this script, please write the correct path.
# gs_file is the source of data.
gs_file = open("C:/Users/Yongda Huang/Desktop/six-work/11-7/gs/11-7-gs", "r")
# output1.dat includes the 2 parameters and 1 minimum value.
output = open("C:/Users/Yongda Huang/Desktop/six-work/11-7/gs/output1.dat", "w")

temp0 = []
for line in gs_file.readlines():
    temp0.append(line.split())
row_number = len(temp0)
column_number = len(temp0[0])
# print(row_number, column_number)
temp1 = np.zeros((row_number, 4))
for i in range(row_number):
    n1 = 0
    for j in [1, 2, 10, 12]:
        temp1[i][n1] = temp0[i][j]
        n1 = n1 + 1
#temp1 is the first data set.
print(temp1)
energy_data_center = temp1[4][2]

for i in range(len(temp1)):
    temp1[i][2] = temp1[i][2] - 0


# 利用高斯分布产生能量的数值
data_number = 4000#chose then number of the data


temp2 = np.zeros((row_number, data_number))
Coefficient_F = np.zeros((data_number, 6)) #15 is the size of coefficients
root = np.zeros((data_number, 4))
# QMC results fitting

for i in range(len(temp1)):
    mu = temp1[i][2]
    sigma = temp1[i][3] #use the data with true error bar
    data_array = np.random.normal(mu, sigma, data_number) #you could set the sigma according to request.
    for a in range(data_number):
        temp2[i][a] = data_array[a]

def solve_solution(x: np.ndarray, y: np.ndarray, r: np.ndarray):
    a = np.block(
        [[x ** 2], [y ** 2], [x * y], [x], [y], [np.ones(len(x))]]).T
    b = r
    p, res, rnk, s = scipy.linalg.lstsq(a, b)
    return p, res, rnk, s, a, b


x = np.zeros(7)
y = np.zeros(7)
r = np.zeros(7)
# print(x)
for i in range(len(x)):
    x[i] = temp1[i][0]
    y[i] = temp1[i][1]

n2 = 0
for j in range(data_number):
    for i in range(len(temp2)):
        r[i] = temp2[i][j]
    p, res, rnk, s, a1, b1 = solve_solution(x, y, r)
    Coefficient_F[j] = p
    n2 = n2 + 1
print(n2)
print('The coefficient of the function')
print(Coefficient_F)
print('-----------------------------------')
print('the process of finding the global minimum')
temp3 = np.zeros(6)
temp4 = np.zeros((len(Coefficient_F), 2))
temp_min = np.zeros(len(Coefficient_F))
for i in range(len(Coefficient_F)):
    #print(i)
    for j in range(len(Coefficient_F[i])):
        temp3[j] = Coefficient_F[i][j]
    #print(temp3)
    c = temp3
    def f(x):
      f1 = c[0] * x[0] * x[0] + c[1] * x[1] * x[1] + c[2] * x[0] * x[1] + c[3] * x[0] + c[4] * x[1] + c[5]
      return f1

    xopt = scipy.optimize.fmin(f, np.array([6, 8]), maxiter=1000, maxfun=1000)
    temp_min[i] = f(xopt)
    temp4[i] = xopt
    print(temp_min[i])
print('-------------------------------------')
print('The 2 parameters of the minimum of the function ')
print(temp4)

temp_e = []
for i in range(len(temp4)):
    for j in range(len(temp4[i])):
        if temp4[i][j] in [6, 8]:
            a = i
            temp_e.append(a)
temp_e_1 = list(set(temp_e))
print('show the line number with problems')
print(temp_e_1) #show the line number with problems
if len(temp_e_1) == 0:
    print("Good!! That is what we need!")
temp5 = np.delete(temp4, temp_e_1, axis=0)
Coefficient_F_c = np.delete(Coefficient_F, temp_e_1, axis=0)
temp_min_f = np.delete(temp_min, temp_e_1, axis=0)
print('-------------------------------------------')
print('The final value (Here we remove the unconvergent data):')
print(temp5) # show the parameters of min
x = []
y = []
for i in range(len(temp5)):
    x.append(temp5[i][0])
    y.append(temp5[i][1])
x1 = np.array(x)
y1 = np.array(y)
print(x1.mean(), x1.std())
print(y1.mean(), y1.std())
print('--------------------------------------------')
for i in range(len(temp5)):
    for j in range(len(temp5[i])):
        output.writelines(str(temp5[i][j])+' ')
    output.writelines(str(temp_min_f[i]))
    output.writelines('\n')
output.close()
for i in range(len(temp5)):
    for j in range(len(temp5[i])):
        if temp5[i][j] < 0:
            print('Error, parameters are smaller than zero. There must be something wrong about the data.')
print('the mean and error of minimum')
print(temp_min_f.mean(), temp_min_f.std())