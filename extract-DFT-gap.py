import numpy as np
import scipy
from scipy.linalg import lstsq
from scipy.optimize import fmin
np.set_printoptions(precision=16)
#This script can plot the all parabola automatically.
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pylab as mtp

temp_path = []
temp_path1 = []
number_file = 640
output = open("C:/Users/Yongda Huang/Desktop/extract-DFT-data/gap-f.dat", "w")

for i in range(number_file):
    temp_path.append('C:/Users/Yongda Huang/Desktop/extract-DFT-data/DFT-gap-f/' + str(i+2) + '/pw/P-1.in')
    temp_path1.append('C:/Users/Yongda Huang/Desktop/extract-DFT-data/DFT-gap-f/' + str(i+2) + '/pw/out')
#+ str(j) + '/'
for d in range(number_file):
    J_file = open(temp_path[d], "r")
    J_file1 = open(temp_path1[d], "r")
    temp1 = []
    temp2 = []
    for line in J_file.readlines():
        temp1.append(line.split())
    for line in J_file1.readlines():
        temp2.append(line.split())
    n = 0
    #print(temp1)
    print(temp2)
    for a in range(len(temp1)):
        for b in range(len(temp1[a])):
            if temp1[a][b] == 'CELL_PARAMETERS':
                output.writelines(str(temp1[a+1][0]) + ' ' + str(temp1[a+2][1]) + ' ' + str(temp2[10]) + ' ' + str(temp2[11]) + '\n' )
                n = n + 1