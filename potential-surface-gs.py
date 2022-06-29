from support import fitting_function
from support import getyv
from support import getexc
import numpy as np
np.set_printoptions(precision=16)
output = open('C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/22-cell/potential-gs-ab.dat', "w")
output1 = open('C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/22-cell/yv.dat', "w")
Coefficient_gs = []
root_scipy_gs = []
root_analy_gs = []
#the roort_f_gs file include the four parameters and minimum value.
root_f_gs = []

Coefficient_gs, root_scipy_gs, root_analy_gs, root_f_gs \
    = fitting_function('C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/22-cell/gs',
                       'C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/22-cell/output_gs', 100)
#print(root_scipy_gs)
#print(root_analy_gs)
#please input the (a, b) mesh
#6.18102	8.78637 0.780072549641792 3.9775382136835606
#a = np.array([6.18102])
#b = np.array([8.78637])
a = np.arange(5.9, 6.60, 0.05) #be careful of the unit
b = np.arange(8.25, 9.25, 0.05)
#a = np.array([6.16695, 6.33574, 6.25135, 6.25135, 6.34981, 6.15289, 6.25135])
#b = np.array([8.80592, 8.57132, 8.88411, 8.49313, 8.82547, 8.55177, 8.68862])
n = 0
total_yv = []
total_value = []
z = []
y = []
z = []
for i in range(len(a)):
     #   j = i #if you want to calculate the points, use this line.
    for j in range(len(b)):
        yv = []
        output.writelines(str(a[i]) + ' ')
        output.writelines(str(b[j]) + ' ')
        print(Coefficient_gs)
        yv, minimum = getyv(a[i], b[j], Coefficient_gs)
        n = n + 1
        for i1 in range(len(yv)):
            output1.writelines(str(yv[i1]) + '\n')
        #print(yv)
        y = []
        z = []
        for i1 in range(len(yv)):
            y.append(yv[i1][0])
            z.append(yv[i1][1])
        y_coord = np.array(y)
        z_coord = np.array(z)
        y_coord_F = y_coord.mean()#/b[j]
        z_coord_F = z_coord.mean()#/37.79452266
        std_y = y_coord.std()
        std_z = z_coord.std()
        print(y_coord_F, z_coord_F)
        print('------------------------------------------------------')
        temp_min = np.array(minimum)
        print('------------------------------------------------------')
        for c in range(len(yv)):
            value = getexc(a[i], b[j], yv[c][0], yv[c][1], Coefficient_gs)
            temp = np.array(value)
        Z = np.array(temp)
        u = Z.mean()
        std = Z.std()
        print('The mean and standard deviation')
        print(u, ' ', std)
        output.writelines(str(y_coord_F)+ ' '+str(std_y)+ ' '+ str(z_coord_F)+' '+str(std_z)+' '+str(u) + ' ' + str(std) + ' ')
        output.writelines('\n')


output.close()

