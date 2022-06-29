import numpy as np
import scipy
from scipy.linalg import lstsq
from scipy.optimize import fmin
from get_gap_support import get_gap

output = open("C:/Users/Yongda Huang/Desktop/six-work/final-gap-3/gap", "w")
output1 = open("C:/Users/Yongda Huang/Desktop/six-work/final-gap-3/gap-1", "w")
gap1 = []
gap1 = get_gap("C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/11-cell/potential-gs-ab.dat",
               "C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/11-cell/potential-exc-ab.dat",
               "C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/11-cell/gap-mesh")
#print(gap1)gap-mesh is output file, so do not worry.
gap2 = []
gap2 = get_gap("C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/22-cell/potential-gs-ab.dat",
               "C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/22-cell/potential-exc-ab.dat",
               "C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/22-cell/gap-mesh")
#print(gap2)

#print(gap3)
gap4 = []
gap4 = get_gap("C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/16-cell/potential-gs-ab.dat",
               "C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/16-cell/potential-exc-ab.dat",
               "C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/16-cell/gap-mesh")
#print(gap4)
gap5 = []
gap5 = get_gap("C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/32-cell/potential-gs-ab.dat",
               "C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/32-cell/potential-exc-ab.dat",
               "C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/32-cell/gap-mesh")

data_number = 1
x_axis = [ 1/16, 1/22, 1/32]
temp16 = np.empty((len(gap1), data_number))
temp22 = np.empty((len(gap1), data_number))
temp32 = np.empty((len(gap1), data_number))

for i1 in range(len(temp16)):
    temp16[i1] = np.random.normal(gap4[i1][2], 0, data_number)
    temp22[i1] = np.random.normal(gap2[i1][2], 0, data_number)
    temp32[i1] = np.random.normal(gap5[i1][2], 0, data_number)

#print(temp11)
slope = np.arange(-11.100000000000007, -11, 0.1)
#print(slope)
least_square = []
error = []
gap_extrapolation = []
for s in slope:
    gap_extrapolation_2 = []
    error1 = []
    least_square_2 = 0
    print('---------------------------')
    for i in range(len(temp16)):
        gap_extrapolation_1 = []
        least_square_1 = 0
        print(i)
        for b in range(data_number):
            for c in range(data_number):
                for d in range(data_number):
                    b, c, d = int(b), int(c), int(d)
                        #print(i, a, b, c, d)
                    X = np.array(x_axis)
                        #print(type(i), type(a))
                    Y = np.array([ temp16[i][b], temp22[i][c], temp32[i][d]])

                    def f(i):
                        return ((Y - (s * X + i)) ** 2).sum()  # 这里面返回的是least square，但是因变量是纵轴的截距。

                    a_1 = scipy.optimize.fsolve(f, x0=2.5)
                    gap_energy = a_1[0]
                        #print(gap_energy, f(a))
                    least_square_1 = f(a_1) + least_square_1
                    gap_extrapolation_1.append(gap_energy)
        gap_extrapolation_2.append(np.array(gap_extrapolation_1).mean())
        error1.append(np.array(gap_extrapolation_1).std())
        #print(gap_extrapolation_2)

    #print(len(gap_extrapolation_1))
    gap_extrapolation.append(gap_extrapolation_2)
    error.append(error1)
    least_square.append(least_square_1)
print('----------------------------------------')
#print(gap_extrapolation_2)
print(gap_extrapolation)
print(error)
gap_extrapolation_f = np.array(gap_extrapolation)
least_square_f = np.array(least_square)
print('----------------------------------------')
print(least_square_f)
print(gap_extrapolation_f)
print('----------------------------------------')
minvalue = least_square_f.min()
minindex = least_square_f.argmin()
print(minindex)
print(slope[minindex])
print(gap_extrapolation_f[minindex])
print(error[minindex])
for k in range(len(gap_extrapolation_f[minindex])):
    output.writelines(str(gap_extrapolation_f[minindex][k]) + ' ' + str(error[minindex][k]) + '\n')

for i in range(len(gap1)):
    #output.writelines(str(gap1[i][0]) + ' ' + str(gap1[i][1]) + ' ' + str(a) + ' ' + str(error) + '\n')
    output1.writelines(str(gap1[i][0])+' '+str(gap1[i][1]) + ' ' + '\n' +
                       str(1/11) + ' ' + str(gap1[i][2]) + ' ' + str(gap1[i][3]) + ' ' + '\n' +
                       str(1/22) + ' ' + str(gap2[i][2]) + ' ' + str(gap2[i][3]) + ' ' + '\n' +
                       str(1/16) + ' ' + str(gap4[i][2]) + ' ' + str(gap4[i][3]) + ' ' + '\n' +
                       str(1/32) + ' ' + str(gap5[i][2]) + ' ' + str(gap5[i][3]) + ' ' + '\n' + '\n')