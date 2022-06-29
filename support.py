import numpy as np
import scipy
from scipy.linalg import lstsq
from scipy.optimize import fmin
from pylab import plot, show
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def fitting_function(par1, par2, par3):
    gs_file = open(par1, "r")
    output = open(par2, "w")
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

    energy_data_center = temp1[16][4]

    for i in range(len(temp1)):
        temp1[i][4] = temp1[i][4] - 0  # energy_data_center, this part is different from fitting-function

    # 利用高斯分布产生能量的数值
    data_number = par3  # chose then number of the data

    temp2 = np.zeros((row_number, data_number))
    Coefficient_F = np.zeros((data_number, 15))  # 15 is the size of coefficients
    root = np.zeros((data_number, 4))
    # QMC results fitting

    for i in range(len(temp1)):
        mu = temp1[i][4]
        sigma = temp1[i][5]  # use the data with true error bar
        data_array = np.random.normal(mu, sigma, data_number)  # you could set the sigma according to request.
        for a in range(data_number):
            temp2[i][a] = data_array[a]

    def solve_solution(x: np.ndarray, y: np.ndarray, z: np.ndarray, v: np.ndarray, r: np.ndarray):
        a = np.block(
            [[x ** 2], [y ** 2], [z ** 2], [v ** 2], [x * y], [x * z], [x * v], [y * z], [y * v], [z * v], [x], [y],
             [z],
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
        # Judge whether the equations have solutions? to be continuned,,,
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
        # print(matrix1)
        Eigenv, Eigenw = np.linalg.eig(matrix1)
        matrix2 = np.linalg.inv(matrix1)
        matrix3 = np.array([-p[10] / 2, -p[11] / 2, -p[12] / 2, -p[13] / 2])
        matrix4 = np.matmul(matrix2, matrix3)
        # print(matrix4)
        if np.min(Eigenv) > 0:
            Coefficient_F[j] = p
            root[j] = matrix4
            n2 = n2 + 1
    print(n2)
    print('THe coefficient of the function')
    print(Coefficient_F)
    print('-----------------------------------')
    print('the process of finding the global minmum')
    temp3 = np.zeros(15)
    temp4 = np.zeros((len(Coefficient_F), 4))
    temp_min = np.zeros(len(Coefficient_F))
    for i in range(len(Coefficient_F)):
        # print(i)
        for j in range(len(Coefficient_F[i])):
            temp3[j] = Coefficient_F[i][j]
        # print(temp3)
        c = temp3

        def f(x):
            f1 = c[0] * x[0] * x[0] + c[1] * x[1] * x[1] + c[2] * x[2] * x[2] + c[3] * x[3] * x[3] + c[4] * x[0] * x[
                1] + c[5] * x[0] * x[2] + c[6] * x[0] * x[3] + c[7] * x[1] * x[2] + c[8] * x[1] * x[3] + c[9] * x[2] * \
                 x[3] + c[10] * x[0] + c[11] * x[1] + c[12] * x[2] + c[13] * x[3] + c[14]
            return f1

        xopt = scipy.optimize.fmin(f, np.array([6, 8, 0.7, 4.0]), maxiter=1000, maxfun=1000)
        temp_min[i] = f(xopt)
        temp4[i] = xopt
        # print(temp_min[i])
    print('-------------------------------------')
    print('The 4 parameters of the minimum of the function ')
    print(temp4)
    print('the 4 parameters calculated analytically')
    print(root)
    # print(root)
    # delete the unconvergent data
    temp_e = []
    for i in range(len(temp4)):
        for j in range(len(temp4[i])):
            if temp4[i][j] in [6, 8, 0.7, 4.0]:
                a = i
                temp_e.append(a)
    temp_e_1 = list(set(temp_e))
    print('show the line number with problems')
    print(temp_e_1)  # show the line number with problems
    if len(temp_e_1) == 0:
        print("Good!! That is what we need!")
    temp5 = np.delete(temp4, temp_e_1, axis=0)
    Coefficient_F_c = np.delete(Coefficient_F, temp_e_1, axis=0)
    temp_min_f = np.delete(temp_min, temp_e_1, axis=0)
    print('-------------------------------------------')
    print('The final value (Here we remove the unconvergent data):')
    print(temp5)  # show the parameters of min
    print('--------------------------------------------')
    print('output the final data')
    # please input your path. Here is all data.
    for i in range(len(temp5)):
        for j in range(len(temp5[i])):
            output.writelines(str(temp5[i][j]) + ' ')
        output.writelines(str(temp_min_f[i]))
        output.writelines('\n')
    output.close()
    for i in range(len(temp5)):
        for j in range(len(temp5[i])):
            if temp5[i][j] < 0:
                print('Error, parameters are smaller than zero. There must be something wrong about the data.')
    # Coefficient_F is the list including final coefficients.
    # temp4 is the list including the parameters obtained by scipy.
    # root is the list including the parameters obtained by analytical solution.
    # temp5 is the list including the parameters excluding the unconvergent data.
    return Coefficient_F_c, temp4, root, temp5


def getyv(x, y, Coefficient):
    temp3 = np.zeros(15)
    temp4 = np.zeros((len(Coefficient), 2))
    temp_min = np.zeros(len(Coefficient))
    for i in range(len(Coefficient)):
        # print(i)
        for j in range(len(Coefficient[i])):
            temp3[j] = Coefficient[i][j]
        # print(temp3)
        c = temp3

        def f(z):
            f1 = c[0] * x * x + c[1] * y * y + c[2] * z[0] * z[0] + c[3] * z[1] * z[1] + c[4] * x * y + c[5] * x * z[
                0] + c[6] * x * z[1] + c[7] * y * z[0] + c[8] * y * z[1] + c[9] * z[0] * z[1] + c[10] * x + c[11] * y + \
                 c[12] * z[0] + c[13] * z[1] + c[14]
            return f1

        xopt = scipy.optimize.fmin(f, np.array([0.75, 3.9]), maxiter=10000, maxfun=10000)
        temp_min[i] = f(xopt)
        temp4[i] = xopt
    return temp4, temp_min


# here Coefficient is a list or ndarry. a, b, y, v is number.
def getab (y, v, Coefficient):
    temp3 = np.zeros(15)
    temp4 = np.zeros((len(Coefficient), 2))
    temp_min = np.zeros(len(Coefficient))
    for i in range(len(Coefficient)):
        # print(i)
        for j in range(len(Coefficient[i])):
            temp3[j] = Coefficient[i][j]
        # print(temp3)
        c = temp3

        def f(x):
            f1 = c[0] * x[0] * x[0] + c[1] * x[1] * x[1] + c[2] * y * y + c[3] * v * v + c[4] * x[0] * x[1] + c[5] * x[
                0] * y + c[6] * x[0] * v + c[7] * x[1] * y + c[8] * x[1] * v + c[9] * y * v + c[10] * x[0] + c[11] * x[
                     1] + c[12] * y + c[13] * v + c[14]
            return f1

        xopt = scipy.optimize.fmin(f, np.array([6, 8]), maxiter=1000, maxfun=1000)
        temp_min[i] = f(xopt)
        temp4[i] = xopt
    return temp4, temp_min


def getexc(a, b, y, v, Coefficient):
    temp3 = np.zeros(15)
    temp_f = np.zeros(len(Coefficient))
    for i in range(len(Coefficient)):
        c = []
        for j in range(len(Coefficient[i])):
            temp3[j] = Coefficient[i][j]
        c = temp3
        f1 = c[0] * a * a + c[1] * b * b + c[2] * y * y + c[3] * v * v + c[4] * a * b + c[5] * a * y + c[6] * a * v + c[
            7] * b * y + c[8] * b * v + c[9] * y * v + c[10] * a + c[11] * b + c[12] * y + c[13] * v + c[14]
        temp_f[i] = f1
    return temp_f

def get_energy(a, b, y, v, Coefficient):
    temp3 = np.zeros(15)
    temp_f = np.zeros(len(Coefficient))
    for i in range(len(Coefficient)):
        c = []
        for j in range(len(Coefficient[i])):
            temp3[j] = Coefficient[i][j]
        c = temp3
        f1 = c[0] * a * a + c[1] * b * b + c[2] * y * y + c[3] * v * v + c[4] * a * b + c[5] * a * y + c[6] * a * v + c[
            7] * b * y + c[8] * b * v + c[9] * y * v + c[10] * a + c[11] * b + c[12] * y + c[13] * v + c[14]
        temp_f[i] = f1
    return temp_f
