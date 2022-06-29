import numpy as np
import scipy
from scipy.linalg import lstsq
from scipy.optimize import fmin
from pylab import plot, show
import matplotlib.pyplot as plt
np.set_printoptions(precision=16)
from mpl_toolkits.mplot3d import Axes3D
#--------------------------------------------------
'''
This script can fit the 4-D function and get the minimum value of the function. 
The script can read the gs straightly. 
The final output file includes the 4 parameters and minimum value.
And it can plot 12 parabola without error bar.
'''
#--------------------------------------------------
#Before use this script, please write the correct path.
# gs_file is the source of data.
gs_file = open("C:/Users/Yongda Huang/Desktop/six-work/22-cell-diagonal-more-walkers/exc/exc", "r")
# output1.dat includes the 4 parameters and 1 minimum value.
output = open("C:/Users/Yongda Huang/Desktop/six-work/22-cell-diagonal-more-walkers/exc/output1.dat", "w")
#This output file output_f_F.dat is the data of 12 parabola with error bar.
output_f_F = open("C:/Users/Yongda Huang/Desktop/six-work/22-cell-diagonal-more-walkers/exc/output_f_F.dat", "w")
temp0 = []
for line in gs_file.readlines():
    temp0.append(line.split())
row_number = len(temp0)
column_number = len(temp0[0])
# print(row_number, column_number)
temp1 = np.zeros((row_number, 6))
for i in range(row_number):
    n1 = 0
    for j in [1, 2, 3, 4, 10, 12]:
        temp1[i][n1] = temp0[i][j]
        n1 = n1 + 1
#temp1 is the first data set.

energy_data_center = temp1[16][4]

for i in range(len(temp1)):
    temp1[i][4] = temp1[i][4] - 0


# 利用高斯分布产生能量的数值
data_number = 4000#chose then number of the data


temp2 = np.zeros((row_number, data_number))
Coefficient_F = np.zeros((data_number, 15)) #15 is the size of coefficients
root = np.zeros((data_number, 4))
# QMC results fitting

for i in range(len(temp1)):
    mu = temp1[i][4]
    #squence = []
    sigma = temp1[i][5] #use the data with true error bar
    data_array = np.random.normal(mu, sigma, data_number) #you could set the sigma according to request.
    for a in range(data_number):
        temp2[i][a] = data_array[a]
'''
# DFT results fitting, if you only want to use DFT data for fitting, enable this part.
for i in range(len(temp1_DFT)):
    mu = temp1_DFT[i][4]
    #sigma = temp1[i][5]
    #if you want to use DFT value, please set the second parameter to sigma=0
    data_array = np.random.normal(mu, 0, data_number)
    for a in range(data_number):
        temp2[i][a] = data_array[a]
'''

def solve_solution(x: np.ndarray, y: np.ndarray, z: np.ndarray, v: np.ndarray, r: np.ndarray):
    a = np.block(
        [[x ** 2], [y ** 2], [z ** 2], [v ** 2], [x * y], [x * z], [x * v], [y * z], [y * v], [z * v], [x], [y], [z],
         [v],
         [np.ones(len(x))]]).T
    b = r
    p, res, rnk, s = scipy.linalg.lstsq(a, b)
    return p, res, rnk, s, a, b


x = np.zeros(25)
y = np.zeros(25)
z = np.zeros(25)
v = np.zeros(25)
r = np.zeros(25)
# print(x)
for i in range(len(x)):
    x[i] = temp1[i][0]
    y[i] = temp1[i][1]
    z[i] = temp1[i][2]
    v[i] = temp1[i][3]
#    r[i] = temp1[i][4]
n2 = 0
for j in range(data_number):
    for i in range(len(temp2)):
        r[i] = temp2[i][j]
    p, res, rnk, s, a1, b1 = solve_solution(x, y, z, v, r)
    #Judge whether the equations have solutions? to be continuned,,,
    n10 = np.linalg.matrix_rank(a1)
    a3 = np.c_[a1, b1]
    n12 = np.linalg.matrix_rank(a3)
    matrix1 = np.zeros((4, 4))
    matrix1[0][0] = p[0]
    matrix1[0][1] = p[4] / 2
    matrix1[0][2] = p[5] / 2
    matrix1[0][3] = p[6] / 2
    matrix1[1][0] = p[4] / 2
    matrix1[1][1] = p[1]
    matrix1[1][2] = p[7] / 2
    matrix1[1][3] = p[8] / 2
    matrix1[2][0] = p[5] / 2
    matrix1[2][1] = p[7] / 2
    matrix1[2][2] = p[2]
    matrix1[2][3] = p[9] / 2
    matrix1[3][0] = p[6] / 2
    matrix1[3][1] = p[8] / 2
    matrix1[3][2] = p[9] / 2
    matrix1[3][3] = p[3]
    #print(matrix1)
    Eigenv, Eigenw = np.linalg.eig(matrix1)
    matrix2 = np.linalg.inv(matrix1)
    matrix3 = np.array([-p[10]/2, -p[11]/2, -p[12]/2, -p[13]/2])
    matrix4 = np.matmul(matrix2, matrix3)
    #print(matrix4)
    print(Eigenv)
    if np.min(Eigenv) > 0:
        Coefficient_F[j] = p
        root[j] = matrix4
        n2 = n2 + 1
print(n2)
print('The coefficient of the function')
print(Coefficient_F)
print('-----------------------------------')
print('the process of finding the global minmum')
temp3 = np.zeros(15)
temp4 = np.zeros((len(Coefficient_F), 4))
temp_min = np.zeros(len(Coefficient_F))
for i in range(len(Coefficient_F)):
    #print(i)
    for j in range(len(Coefficient_F[i])):
        temp3[j] = Coefficient_F[i][j]
    #print(temp3)
    c = temp3

    def f(x):
      f1 = c[0] * x[0] * x[0] + c[1] * x[1] * x[1] + c[2] * x[2] * x[2] + c[3] * x[3] * x[3] + c[4] * x[0] * x[1] + c[5] * x[0] * x[2] + c[6] * x[0] * x[3] + c[7] * x[1] * x[2] + c[8] * x[1] * x[3] + c[9] * x[2] * x[3] + c[10] * x[0] + c[11] * x[1] + c[12] * x[2] + c[13] * x[3] + c[14]
      return f1

    xopt = scipy.optimize.fmin(f, np.array([6, 8, 0.75, 3.9]), maxiter=1000, maxfun=1000)
    temp_min[i] = f(xopt)
    temp4[i] = xopt
    print(temp_min[i])
print('-------------------------------------')
print('The 4 parameters of the minimum of the function ')
print(temp4)
print('the 4 parameters calculated analytically')
print(root)
#print(root)
#delete the unconvergent data
temp_e = []
for i in range(len(temp4)):
    for j in range(len(temp4[i])):
        if temp4[i][j] in [6, 8, 0.75, 3.9]:
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
z = []
v = []
for i in range(len(temp5)):
    x.append(temp5[i][0])
    y.append(temp5[i][1])
    z.append(temp5[i][2])
    v.append(temp5[i][3])
x1 = np.array(x)
y1 = np.array(y)
z1 = np.array(z)
v1 = np.array(v)
print(x1.mean(), x1.std())
print(y1.mean(), y1.std())
print(z1.mean(), z1.std())
print(v1.mean(), v1.std())
print('--------------------------------------------')
#print('output the final data')
#please input your path. Here is all data.

for i in range(len(temp5)):
    for j in range(len(temp5[i])):
        output.writelines(str(temp5[i][j])+' ')
    output.writelines(str(temp_min_f[i]))
    output.writelines('\n')
output.close()
#print(len(Coefficient_F_c))
#print(len(temp5))
for i in range(len(temp5)):
    for j in range(len(temp5[i])):
        if temp5[i][j] < 0:
            print('Error, parameters are smaller than zero. There must be something wrong about the data.')
#project to specific 1,16,17 planes
#print(temp1)
#print(temp1[0], temp1[15])
vector = np.zeros((12, 4))
#print(vector)
for i in range(4):
    vector[0][i] = temp1[15][i] - temp1[0][i]
    vector[1][i] = temp1[14][i] - temp1[1][i]
    vector[2][i] = temp1[13][i] - temp1[2][i]
    vector[3][i] = temp1[12][i] - temp1[3][i]
    vector[4][i] = temp1[11][i] - temp1[4][i]
    vector[5][i] = temp1[10][i] - temp1[5][i]
    vector[6][i] = temp1[9][i] - temp1[6][i]
    vector[7][i] = temp1[8][i] - temp1[7][i]
    vector[8][i] = temp1[17][i] - temp1[18][i]
    vector[9][i] = temp1[19][i] - temp1[20][i]
    vector[10][i] = temp1[21][i] - temp1[22][i]
    vector[11][i] = temp1[23][i] - temp1[24][i]
X = []
Y = []
Z = []
V = []
t = np.linspace(0, 1, 50)
start =[temp1[0], temp1[1], temp1[2], temp1[3], temp1[4], temp1[5], temp1[6], temp1[7], temp1[18], temp1[20], temp1[22], temp1[24]]
#print(start)
for i in range(len(start)):
    X1 = []
    Y1 = []
    Z1 = []
    V1 = []
    for j in t:
        X1.append(start[i][0] + j*vector[i][0])
        Y1.append(start[i][1] + j*vector[i][1])
        Z1.append(start[i][2] + j*vector[i][2])
        V1.append(start[i][3] + j*vector[i][3])
    X.append(X1)
    Y.append(Y1)
    Z.append(Z1)
    V.append(V1)

f_total = []
for i in range(len(Coefficient_F_c)):
    for j in range(len(X)):
        #print(j)
        f_xyzv = []
        for v in range(len(X[j])):
            f_xyzv.append(Coefficient_F_c[i][0]*X[j][v]*X[j][v]+Coefficient_F_c[i][1]*Y[j][v]*Y[j][v]+Coefficient_F_c[i][2]*Z[j][v]*Z[j][v]+Coefficient_F_c[i][3]*V[j][v]*V[j][v]+Coefficient_F_c[i][4]*X[j][v]*Y[j][v]+Coefficient_F_c[i][5]*X[j][v]*Z[j][v]+Coefficient_F_c[i][6]*X[j][v]*V[j][v]+Coefficient_F_c[i][7]*Y[j][v]*Z[j][v]+Coefficient_F_c[i][8]*Y[j][v]*V[j][v]+Coefficient_F_c[i][9]*Z[j][v]*V[j][v]+Coefficient_F_c[i][10]*X[j][v]+Coefficient_F_c[i][11]*Y[j][v]+Coefficient_F_c[i][12]*Z[j][v]+Coefficient_F_c[i][13]*V[j][v]+Coefficient_F_c[i][14])
        #plt.plot(t, f_xyzv)
        f_total.append(f_xyzv)
        #plt.xlabel('x')
        #plt.ylabel('f_x')
        #if you would like to plot all parabola with corresponding coefficients, use this.
        #plt.show()

f_F = []
np.array(f_total)
for i in range(12):
    c = np.zeros(50)
    for j in range(i, len(Coefficient_F_c)*12, 12):
        c = c + np.array(f_total[j])
    f_F.append(c)
#print(f_F[0])
#print(f_F[1])
print('----------------------------------')
print('error bar')
error = []
c = []
for i in range(12):
    b = []
    for k in range(50):
        a = []
        for j in range(i, len(Coefficient_F_c)*12, 12):
            a.append(f_total[j][k])
        b.append(a)
    c.append(b)
print('-------------------------------')
d = np.array(c)
#print(d)
for i in range(len(d)):
    for j in range(len(d[i])):
        #print(d[i][j])
        error.append(d[i][j].std())
#print(error)
print('-----------------------------------')
print('Here we show the numer of parabola. it should be 12. Otherwise the calculation is wrong.')
print(len(f_F))
print('-----------------------------------')
print('Output the data of 12 parabola')
n = 0
for i in range(len(f_F)):
    for j in range(len(f_F[i])):
        output_f_F.writelines(str(t[j]) + ' ')
        output_f_F.writelines(str(f_F[i][j]/len(Coefficient_F_c)) + ' ' + str(error[n]))
        output_f_F.writelines('\n')
        n = n + 1
    output_f_F.writelines('\n')
output_f_F.close()
print('-----------------------------------')
print('plot the parabola without error bar')
z = np.zeros(len(t))
plt.figure(figsize=(40, 30))
for i in range(12):
    plt.subplot(4, 3, i+1)
    plt.plot(t, f_F[i]/len(Coefficient_F_c))
    #print(len(f_F[i]))
    #plt.plot(t, z)
    plt.title(i+1)
    plt.xlabel('t')
    plt.ylabel('f')
#plt.show()
plt.savefig('C:/Users/Yongda Huang/Desktop/six-work/22-cell-diagonal-more-walkers/exc/')
#-----------------
print('the mean and error of minimum')
print(temp_min_f.mean(), temp_min_f.std())
#-----------------------------------------
print('print energy of arbitrary point')
#6.1 8.9  0.7981353  3.96414376
for i in range(1):
    X = []
    Y = []
    Z = []
    V = []
    X.append(6.1)
    Y.append(8.9)
    Z.append(0.7981353)
    V.append(3.96414376)

f_xyzv = []
f_total_1 = []
for i in range(len(Coefficient_F_c)):
        f_xyzv.append(Coefficient_F_c[i][0]*X[0]*X[0]+Coefficient_F_c[i][1]*Y[0]*Y[0]+Coefficient_F_c[i][2]*Z[0]*Z[0]+Coefficient_F_c[i][3]*V[0]*V[0]+Coefficient_F_c[i][4]*X[0]*Y[0]+Coefficient_F_c[i][5]*X[0]*Z[0]+Coefficient_F_c[i][6]*X[0]*V[0]+Coefficient_F_c[i][7]*Y[0]*Z[0]+Coefficient_F_c[i][8]*Y[0]*V[0]+Coefficient_F_c[i][9]*Z[0]*V[0]+Coefficient_F_c[i][10]*X[0]+Coefficient_F_c[i][11]*Y[0]+Coefficient_F_c[i][12]*Z[0]+Coefficient_F_c[i][13]*V[0]+Coefficient_F_c[i][14])
        f_total_1.append(f_xyzv)
f_arb = np.array(f_total_1)
print(f_arb.mean(), f_arb.std())
#the following parts are useless. But the codes are good.
'''
output_xyzv = open("C:/Users/Yongda Huang/Desktop/second work/fitting/output_xyzv.dat", "w")
for i in range(len(Coefficient_F_c)):
    X = np.linspace(3, 4, 1000)
    Y = np.linspace(4, 5, 1000)
    Z = np.linspace(-1, 1, 1000)
    V = np.linspace(-1, 1, 1000)
    f_xyzv = []
    for j in range(len(X)):
        f_xyzv.append(Coefficient_F_c[i][0]*X[j]*X[j]+Coefficient_F_c[i][1]*Y[j]*Y[j]+Coefficient_F_c[i][2]*Z[j]*Z[j]+Coefficient_F_c[i][3]*V[j]*V[j]+Coefficient_F_c[i][4]*X[j]*Y[j]+Coefficient_F_c[i][5]*X[j]*Z[j]+Coefficient_F_c[i][6]*X[j]*V[j]+Coefficient_F_c[i][7]*Y[j]*Z[j]+Coefficient_F_c[i][8]*Y[j]*V[j]+Coefficient_F_c[i][9]*Z[j]*V[j]+Coefficient_F_c[i][10]*X[j]+Coefficient_F_c[i][11]*Y[j]+Coefficient_F_c[i][12]*Z[j]+Coefficient_F_c[i][13]*V[j]+Coefficient_F_c[i][14])

    for a in range(len(Y)):
        output_xyzv.writelines(str(X[a]) + ' ' + str(Y[a]) + ' ' + str(Z[a]) + ' ' + str(V[a])+' '+str(f_xyzv[a]))
        output_xyzv.writelines('\n')
output_xyzv.close()

#The part is for output. It is not necessary.
#project to x axis. The code can generate picture and data.
for i in range(len(Coefficient_F_c)):
    X = np.linspace(2, 13, 1000) # you can modify the length of X axis.
    f_x = []
    for x in X:
        f_x.append(Coefficient_F_c[i][0]*x*x+Coefficient_F_c[i][10]*x+Coefficient_F_c[i][14])
    print(f_x)
    output_x = open("C:/Users/Yongda Huang/Desktop/second work/fitting/output_x.dat", "w")
    for a in range(len(X)):
        output_x.writelines(str(X[a]) + ' ' + str(f_x[a]))
        output_x.writelines('\n')
    output_x.close()
    plt.plot(X, f_x)
    plt.xlabel('x')
    plt.ylabel('f_x')
    plt.show()
#project to y axis
for i in range(len(Coefficient_F_c)):
    Y = np.linspace(5, 13, 1000)
    f_y = []
    for y in Y:
        f_y.append(Coefficient_F_c[i][1]*y*y+Coefficient_F_c[i][11]*y+Coefficient_F_c[i][14])
    print(f_y)
    output_y = open("C:/Users/Yongda Huang/Desktop/second work/fitting/output_y.dat", "w")
    for a in range(len(Y)):
        output_y.writelines(str(Y[a]) + ' ' + str(f_y[a]))
        output_y.writelines('\n')
    output_y.close()
    plt.plot(Y, f_y)
    plt.xlabel('y')
    plt.ylabel('f_y')
    plt.show()

#project to xy plane. For other planes, the codes are simliar. You only need to change the Coefficient_F.
for i in range(len(Coefficient_F_c)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.linspace(6, 13, 1000)
    Y = np.linspace(5, 13, 1000)
    f_xy = []
    for j in range(len(X)):
        f_xy.append(Coefficient_F_c[i][0]*X[j]*X[j]+Coefficient_F_c[i][1]*Y[j]*Y[j]+Coefficient_F_c[i][4]*X[j]*Y[j]+Coefficient_F_c[i][10]*X[j]+Coefficient_F_c[i][11]*Y[j]+Coefficient_F_c[i][14])
    ax.plot(X, Y, f_xy)
    ax.set_xlabel('x Label')
    ax.set_ylabel('y Label')
    ax.set_zlabel('f_xy Label')
    plt.show()
 
#project to xyz dimension. But it won't show the figure. You can plot it by using Origin or other tools.
for i in range(len(Coefficient_F_c)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.linspace(6, 13, 1000)
    Y = np.linspace(5, 13, 1000)
    Z = np.linspace(-2, 2, 1000)
    f_xyz = []
    j: int
    for j in range(len(X)):
        f_xyz.append(Coefficient_F_c[i][0]*X[j]*X[j]+Coefficient_F_c[i][1]*Y[j]*Y[j]+Coefficient_F_c[i][2]*Z[j]*Z[j]+Coefficient_F_c[i][5]*X[j]*Z[j]+Coefficient_F_c[i][7]*Y[j]*Z[j]+Coefficient_F_c[i][4]*X[j]*Y[j]+Coefficient_F_c[i][10]*X[j]+Coefficient_F_c[i][11]*Y[j]+Coefficient_F_c[i][12]*Z[j]+Coefficient_F_c[i][14])
    output_xyz = open("C:/Users/Yongda Huang/Desktop/second work/fitting/output_xyz.dat", "w")
    for a in range(len(Y)):
        output_xyz.writelines(str(X[a]) + ' ' + str(Y[a]) + ' ' + str(Z[a]) + ' ' + str(f_xyz[a]))
        output_xyz.writelines('\n')
    output_xyz.close()
'''

print('-----------------------------------------------')
print('I think it is done. If not, please contact me')



