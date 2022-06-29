import numpy as np
import scipy.optimize
np.set_printoptions(precision=16)

x_axis = [1/16, 1/22, 1/32]

y_axis_11 = [1.76446, 1.79284, 1.86807, 1.62793,  2.01611,  1.65437, 1.80946]
y_axis_11_error = [0.03406, 0.03766, 0.04895, 0.04227, 0.03346, 0.03765, 0.02986]

y_axis_16 = [2.07734, 2.10318,  2.1294,  1.84591,  2.23383,  1.79138, 2.05347]
y_axis_16_error = [0.05317, 0.05931, 0.06818, 0.07037, 0.04851, 0.06412, 0.05693]

y_axis_22 = [2.19569, 2.25951, 2.27048,  2.04407,  2.46297,  2.07821, 2.17676]
y_axis_22_error = [0.02749, 0.04342, 0.03186, 0.03297, 0.04269, 0.03381, 0.03477]

y_axis_32 = [2.47121,  2.42822,  2.5885,  2.27904,  2.64067,  2.27659,  2.42531]
y_axis_32_error = [0.04673, 0.05074, 0.0485, 0.04718, 0.04945, 0.04715, 0.05047]

data_number = 100
temp16 = np.empty((7, data_number))
temp22 = np.empty((7, data_number))
temp32 = np.empty((7, data_number))

#print(temp)
for i1 in range(len(temp16)):
    temp16[i1] = np.random.normal(y_axis_16[i1], y_axis_16_error[i1], data_number)
    temp22[i1] = np.random.normal(y_axis_22[i1], y_axis_22_error[i1], data_number)
    temp32[i1] = np.random.normal(y_axis_32[i1], y_axis_32_error[i1], data_number)


slope = np.arange(-13, -10, 0.1)
#print(slope)
least_square = []
error = []
gap_extrapolation = []
for s in slope:
    print(s)
    gap_extrapolation_2 = []
    error1 = []
    least_square_2 = 0
    print('---------------------------')
    for i in range(7):
        print(i)
        gap_extrapolation_1 = []
        least_square_1 = 0
        for b in range(data_number):
            for c in range(data_number):
                for d in range(data_number):
                    b, c, d = int(b), int(c), int(d)
                        #print(i, a, b, c, d)
                    X = np.array(x_axis)
                        #print(type(i), type(a))
                    Y = np.array([temp16[i][b], temp22[i][c], temp32[i][d]])

                    def f(i):
                        return ((Y - (s * X + i)) ** 2).sum()  # 这里面返回的是least square，但是因变量是纵轴的截距。

                    a_1 = scipy.optimize.fsolve(f, x0=2.5)
                    gap_energy = a_1[0]
                        #print(gap_energy, f(a_1))
                    least_square_1 = f(a_1) + least_square_1
                        #print(least_square_1)
                    gap_extrapolation_1.append(gap_energy)
        gap_extrapolation_2.append(np.array(gap_extrapolation_1).mean())
        error1.append(np.array(gap_extrapolation_1).std())
        #print(gap_extrapolation_2)

    #print(len(gap_extrapolation_1))
    gap_extrapolation.append(gap_extrapolation_2)
    error.append(error1)
    least_square.append(least_square_1)
    print(least_square_1)
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



