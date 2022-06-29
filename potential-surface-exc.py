from support import fitting_function
from support import getyv
from support import getexc
import numpy as np
output = open('C:/Users/Yongda Huang/Desktop/six-work/DFT/exc/potential-exc-ab.dat', "w")
output1 = open('C:/Users/Yongda Huang/Desktop/six-work/DFT/exc/yv.dat', "w")
Coefficient_exc = []
root_scipy_exc = []
root_analy_exc = []
#the roort_f_gs file include the four parameters and minimum value.
root_f_exc = []

Coefficient_exc, root_scipy_exc, root_analy_exc, root_f_exc \
    = fitting_function('C:/Users/Yongda Huang/Desktop/six-work/DFT/exc/exc',
                       'C:/Users/Yongda Huang/Desktop/six-work/DFT/exc/output_exc', 100)
#please input the (a, b) mesh
a = np.arange(5.9, 6.60, 0.05) #be careful of the unit
b = np.arange(8.25, 9.25, 0.05)
n = 0
total_yv = []
total_value = []
z = []
y = []
z = []
for i in range(len(a)):
    for j in range(len(b)):
        yv = []
        output.writelines(str(a[i]) + ' ')
        output.writelines(str(b[j]) + ' ')
        print(Coefficient_exc)
        yv, minimum = getyv(a[i], b[j], Coefficient_exc)
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
            value = getexc(a[i], b[j], yv[c][0], yv[c][1], Coefficient_exc)
            temp = np.array(value)
        Z = np.array(temp)
        u = Z.mean()
        std = Z.std()
        print('The mean and standard deviation')
        print(u, ' ', std)
        output.writelines(str(y_coord_F)+ ' '+str(std_y)+ ' '+ str(z_coord_F)+' '+str(std_z)+' '+str(u) + ' ' + str(std) + ' ')
        output.writelines('\n')

output.close()