from support import fitting_function
from support import getyv
from support import getexc
import numpy as np
from support7points import fitting_function7
from support7points import get_results
from support import get_energy
from scipy import stats
import scipy
from scipy.linalg import lstsq
from scipy.optimize import fmin
#output for gs
output = open('C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/22-cell/potential-exc-ab.dat', "w")

output11_gs = open('C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/11-cell/potential-gs-ab.dat', "w")
output11_exc = open('C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/11-cell/potential-exc-ab.dat', "w")
output16_gs = open('C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/16-cell/potential-gs-ab.dat', "w")
output16_exc = open('C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/16-cell/potential-exc-ab.dat', "w")
output32_gs = open('C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/32-cell/potential-gs-ab.dat', "w")
output32_exc = open('C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/32-cell/potential-exc-ab.dat', "w")
Coefficient_gs = []
root_scipy_gs = []
root_analy_gs = []
#the roort_f_gs file include the four parameters and minimum value.
root_f_gs = []
#output for exec
Coefficient_exc = []
root_scipy_exc = []
root_analy_exc = []
root_f_exc = []
data_number = 100
Coefficient_gs, root_scipy_gs, root_analy_gs, root_f_gs \
    = fitting_function('C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/22-cell/gs',
                       'C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/22-cell/output_gs', data_number)
Coefficient_exc, root_scipy_exc, root_f_exc \
    = fitting_function7('C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/22-cell/exc',
                       'C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/22-cell/output_exc', data_number)

Coefficient_gs_16 = []
root_scipy_gs_16 = []
root_analy_gs_16 = []
#the roort_f_gs file include the four parameters and minimum value.
root_f_gs_16 = []
#output for exec
Coefficient_exc_16 = []
root_scipy_exc_16 = []
root_analy_exc_16 = []
root_f_exc_16 = []
Coefficient_gs_16, root_scipy_gs_16, root_analy_gs_16, root_f_gs_16 \
    = fitting_function('C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/16-cell/gs',
                       'C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/16-cell/output_gs', data_number)
Coefficient_exc_16, root_scipy_exc_16, root_analy_exc_16, root_f_exc_16 \
    = fitting_function('C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/16-cell/exc',
                       'C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/16-cell/output_exc', data_number)
Coefficient_gs_11 = []
root_scipy_gs_11 = []
root_analy_gs_11 = []
#the roort_f_gs file include the four parameters and minimum value.
root_f_gs_11 = []
#output for exec
Coefficient_exc_11 = []
root_scipy_exc_11 = []
root_analy_exc_11 = []
root_f_exc_11 = []
Coefficient_gs_11, root_scipy_gs_11, root_analy_gs_11, root_f_gs_11 \
    = fitting_function('C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/11-cell/gs',
                       'C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/11-cell/output_gs', data_number)
Coefficient_exc_11, root_scipy_exc_11, root_analy_exc_11, root_f_exc_11 \
    = fitting_function('C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/11-cell/exc',
                       'C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/11-cell/output_exc', data_number)

Coefficient_exc_32, root_scipy_exc_32, root_f_exc_32 \
    = fitting_function7('C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/32-cell/exc',
                       'C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/32-cell/output_exc', data_number)

Coefficient_gs_32, root_scipy_gs_32, root_f_gs_32 \
    = fitting_function7('C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/32-cell/gs',
                       'C:/Users/Yongda Huang/Desktop/six-work/final-gap-2/32-cell/output_gs', data_number)
#please input the (a, b) mesh
a = np.arange(5.9, 6.60, 0.05) #be careful of the unit
b = np.arange(8.25, 9.25, 0.05)
#a = np.array([6.16695, 6.33574, 6.25135, 6.25135, 6.34981, 6.15289, 6.25135])
#b = np.array([8.80592, 8.57132, 8.88411, 8.49313, 8.82547, 8.55177, 8.68862])
n = 0
total_yv = []
total_value = []
z = []
for i in range(len(a)):
    for j in range(len(b)):
    #    j = i
        yv = []
        output.writelines(str(a[i]) + ' ')
        output.writelines(str(b[j]) + ' ')
        output16_gs.writelines(str(a[i]) + ' ')
        output16_gs.writelines(str(b[j]) + ' ')
        output16_exc.writelines(str(a[i]) + ' ')
        output16_exc.writelines(str(b[j]) + ' ')
        output11_gs.writelines(str(a[i]) + ' ')
        output11_gs.writelines(str(b[j]) + ' ')
        output11_exc.writelines(str(a[i]) + ' ')
        output11_exc.writelines(str(b[j]) + ' ')
        output32_gs.writelines(str(a[i]) + ' ')
        output32_gs.writelines(str(b[j]) + ' ')
        output32_exc.writelines(str(a[i]) + ' ')
        output32_exc.writelines(str(b[j]) + ' ')
        #print(Coefficient_gs)
        yv, minimum = getyv(a[i], b[j], Coefficient_gs)
        n = n + 1
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
            value = get_results(a[i], b[j], Coefficient_exc)
            temp = np.array(value)

            value16_gs = get_energy(a[i], b[j], yv[c][0], yv[c][1], Coefficient_gs_16)
            value16_exc = get_energy(a[i], b[j], yv[c][0], yv[c][1], Coefficient_exc_16)
            temp16_gs = np.array(value16_gs)
            temp16_exc = np.array(value16_exc)

            value11_gs = get_energy(a[i], b[j], yv[c][0], yv[c][1], Coefficient_gs_11)
            value11_exc = get_energy(a[i], b[j], yv[c][0], yv[c][1], Coefficient_exc_11)
            temp11_gs = np.array(value11_gs)
            temp11_exc = np.array(value11_exc)

            value32_gs = get_results(a[i], b[j], Coefficient_gs_32)
            value32_exc = get_results(a[i], b[j], Coefficient_exc_32)
            temp32_gs = np.array(value32_gs)
            temp32_exc = np.array(value32_exc)


        Z = np.array(temp)
        u = Z.mean()
        std = Z.std()
        #print('The mean and standard deviation')
        #print(u, ' ', std)
        output.writelines(str(y_coord_F)+ ' '+str(std_y)+ ' '+ str(z_coord_F)+' '+str(std_z)+' '+str(u) + ' ' + str(std) + ' ')
        output.writelines('\n')

        Z_16_gs = np.array(temp16_gs)
        u_16_gs = Z_16_gs.mean()
        std_16_gs = Z_16_gs.std()
        output16_gs.writelines(
            str(y_coord_F) + ' ' + str(std_y) + ' ' + str(z_coord_F) + ' ' + str(std_z) + ' ' + str(u_16_gs) + ' ' + str(
                std_16_gs) + ' ')
        output16_gs.writelines('\n')
        Z_16_exc = np.array(temp16_exc)
        u_16_exc = Z_16_exc.mean()
        std_16_exc = Z_16_exc.std()
        output16_exc.writelines(
            str(y_coord_F) + ' ' + str(std_y) + ' ' + str(z_coord_F) + ' ' + str(std_z) + ' ' + str(u_16_exc) + ' ' + str(
                std_16_exc) + ' ')
        output16_exc.writelines('\n')

        Z_11_gs = np.array(temp11_gs)
        u_11_gs = Z_11_gs.mean()
        std_11_gs = Z_11_gs.std()
        output11_gs.writelines(
            str(y_coord_F) + ' ' + str(std_y) + ' ' + str(z_coord_F) + ' ' + str(std_z) + ' ' + str(u_11_gs) + ' ' + str(
                std_11_gs) + ' ')
        output11_gs.writelines('\n')
        Z_11_exc = np.array(temp11_exc)
        u_11_exc = Z_11_exc.mean()
        std_11_exc = Z_11_exc.std()
        output11_exc.writelines(
            str(y_coord_F) + ' ' + str(std_y) + ' ' + str(z_coord_F) + ' ' + str(std_z) + ' ' + str(u_11_exc) + ' ' + str(
                std_11_exc) + ' ')
        output11_exc.writelines('\n')

        Z_32_gs = np.array(temp32_gs)
        u_32_gs = Z_32_gs.mean()
        std_32_gs = Z_32_gs.std()
        output32_gs.writelines(
            str(y_coord_F) + ' ' + str(std_y) + ' ' + str(z_coord_F) + ' ' + str(std_z) + ' ' + str(u_32_gs) + ' ' + str(
                std_32_gs) + ' ')
        output32_gs.writelines('\n')
        Z_32_exc = np.array(temp32_exc)
        u_32_exc = Z_32_exc.mean()
        std_32_exc = Z_32_exc.std()
        output32_exc.writelines(
            str(y_coord_F) + ' ' + str(std_y) + ' ' + str(z_coord_F) + ' ' + str(std_z) + ' ' + str(u_32_exc) + ' ' + str(
                std_32_exc) + ' ')
        output32_exc.writelines('\n')


output.close()

