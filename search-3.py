import pyscal as pc
import numpy as np
np.set_printoptions(precision=16)

#first we need create the combined POSCAR for P-SiO2.
# This script can help us combine two POSCAR to generate interface. So we do not need VESTA anymore. But only cart coordiante.

# i means the number of files.
for i in range(20):
    structure_file_P = open('C:/Users/Yongda Huang/Desktop/SiO2-substrate/final-5-compare-script-test/test/POSCAR-' + str(i+1) + '-P.vasp', "r")
    structure_file_SiO2 = open('C:/Users/Yongda Huang/Desktop/SiO2-substrate/final-5-compare-script-test/test/POSCAR-' + str(i+1) + '-SiO2.vasp', "r")
    output_P_SiO2 = open('C:/Users/Yongda Huang/Desktop/SiO2-substrate/final-5-compare-script-test/test/POSCAR-' + str(i+1) + '-P-SiO2.vasp-1', "w")
    temp_P = []
    temp_SiO2 = []
    for line in structure_file_P.readlines():
        temp_P.append(line.split())
    for line in structure_file_SiO2.readlines():
        temp_SiO2.append(line.split())
    #print(temp_SiO2)
    target_x = np.sqrt(float(temp_SiO2[2][0])**2 + float(temp_SiO2[2][1])**2)
    #print(float(temp_SiO2[2][0]))
    #print(float(temp_SiO2[2][1]))
    #print(target_x)
    x1 = float(temp_SiO2[2][0])
    y1 = float(temp_SiO2[2][1])
    x2 = target_x
    y2 = 0
    vector1 = np.array([x1, y1])
    vector2 = np.array([x2, y2])
    # here, we create a rotation matrix to transform SiO2 to P-SiO2.
    U1 = np.array([[(x1 * x2 + y1 * y2)/(np.sqrt(x1**2 + y1**2)*np.sqrt(x2**2 + y2**2)), -(x1 * y2 - x2 * y1)/(np.sqrt(x1**2 + y1**2)*np.sqrt(x2**2 + y2**2))],
                   [(x1 * y2 - x2 * y1)/(np.sqrt(x1**2 + y1**2)*np.sqrt(x2**2 + y2**2)), (x1 * x2 + y1 * y2)/(np.sqrt(x1**2 + y1**2)*np.sqrt(x2**2 + y2**2))]])

    #print(U1)
    #print(np.dot(U1, vector1))
    vector3 = np.array([float(temp_SiO2[3][0]), float(temp_SiO2[3][1])])
    vector4 = np.dot(U1, vector3)
    #print(vector4)
    output_P_SiO2.writelines('SiO2-P' + '\n')
    output_P_SiO2.writelines('1.0' + '\n')
    output_P_SiO2.writelines(str(x2) + ' ' + str(y2) + '0' + '\n')
    output_P_SiO2.writelines(str(vector4[0]) + ' ' + str(vector4[1]) + ' '+ '0' + '\n')
    output_P_SiO2.writelines('0' + ' ' + '0' + ' ' + str(temp_SiO2[4][2]) + '\n')
    output_P_SiO2.writelines(str(temp_SiO2[5][0] + ' ' + str(temp_SiO2[5][1] + ' ' + str(temp_P[5][0]) + '\n')))
    output_P_SiO2.writelines(str(temp_SiO2[6][0] + ' ' + str(temp_SiO2[6][1] + ' ' + str(temp_P[6][0]) + '\n')))
    output_P_SiO2.writelines('cartesian' + ' ' + '\n')
    #for a in temp_SiO2:
     #   print(a)
    temp_vector = []
    for a in range(8, len(temp_SiO2)):
        #print(temp_SiO2[a])
        v1 = np.array([float(temp_SiO2[a][0]), float(temp_SiO2[a][1])])
        vector5 = np.dot(U1, v1)
        temp_vector.append([vector5[0], vector5[1], temp_SiO2[a][2]])
    #print(temp_vector)
    #for a in temp_vector:
        #print(a)
    for a in range(len(temp_vector)):
        output_P_SiO2.writelines('  ' + str(temp_vector[a][0]) + '   ' + str(temp_vector[a][1]) + '  ' + str(temp_vector[a][2] + ' ' + '\n'))

    target_x = np.sqrt(float(temp_P[2][0]) ** 2 + float(temp_P[2][1]) ** 2)
    x3 = float(temp_P[2][0])
    y3 = float(temp_P[2][1])
    x4 = target_x
    y4 = 0
    vector1 = np.array([x3, y3])
    vector2 = np.array([x4, y4])
    # here, we creat another rotation from P to P-SiO2.
    U2 = np.array([[(x3 * x4 + y3 * y4) / (np.sqrt(x3 ** 2 + y3 ** 2) * np.sqrt(x4 ** 2 + y4 ** 2)),
                    -(x3 * y4 - x4 * y3) / (np.sqrt(x3 ** 2 + y3 ** 2) * np.sqrt(x4 ** 2 + y4 ** 2))],
                   [(x3 * y4 - x2 * y3) / (np.sqrt(x3 ** 2 + y3 ** 2) * np.sqrt(x4 ** 2 + y4 ** 2)),
                    (x3 * x4 + y3 * y4) / (np.sqrt(x3 ** 2 + y3 ** 2) * np.sqrt(x4 ** 2 + y4 ** 2))]])

    vector3 = np.array([6.26325773400000,  59.07766579000000])
    print(np.dot(U2, vector3))
    temp_vector_P = []
    for a in range(8, len(temp_P)):
        #print(temp_SiO2[a])
        v1 = np.array([float(temp_P[a][0]), float(temp_P[a][1])])
        vector6 = np.dot(U2, v1)
        temp_vector_P.append([vector6[0], vector6[1], temp_P[a][2]])

    for a in range(len(temp_vector_P)):
        output_P_SiO2.writelines('  ' + str(temp_vector_P[a][0]) + '   ' + str(temp_vector_P[a][1]) + '  ' + str(float(temp_vector_P[a][2]) + 7) + ' ' + '\n') # number 7 is the distance paramter between two layers.

    print('I am really smart!')



