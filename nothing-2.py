import os,math
cores = os.cpu_count()
cores = 24
import numpy as np
from nexus import settings,job,run_project
from nexus import generate_physical_system
from nexus import generate_pwscf
from nexus import generate_pw2qmcpack
from nexus import generate_qmcpack,vmc,dmc
from qmcpack_input import nofk,gofr,correlation, jastrow1, jastrow2, jastrow3, section
from vasp import generate_poscar

settings(
        pseudo_dir = './pp',
        results    = '',
        sleep      = 10,
        machine    = 'ws'+str(cores),
        )



pwscf      = '/home/yongda-1/qe/bin/pw.x'
pw2qmcpack ='/home/yongda-1/qe/bin/pw2qmcpack.x'
qmcpack    ='/home/yongda-1/yongda-1/qmcpack-3.12.0-complex/qmcpack-3.12.0/build/bin/qmcpack_complex'
output = open('/home/yongda-1/yongda-1/DFT-substrate-P/new-lattice-mismatch-1/P-strain-1', "w")
#print(temp0)
i1 = 0
shape1 = []
shape2 = []
list1 = np.arange(-6, 7, 1)
for a in list1:
    for b in list1:
        for c in list1:
            for d in list1:
                if np.abs(a*d - b*c) > 1:
                    #print(a, b, c, d)
                    system  = generate_physical_system(
                        units = 'A',                  # Angstrom units
                        axes  = [[4.9494877691957697, 0, 0],     # Cell axes
                                 [-2.4747438845978902,  4.2863821438439098, 0],
                                 [0, 0, 32.6939163208]],
                        elem  = ['Si','Si','Si','Si','Si','Si','Si','Si','Si','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O'],            # Element names
                        posu  = [  [0.8376443532343929,  0.0096178778000012,  0.0500987430948520],
                                   [0.1826456224014148,  0.6737937930069107,  0.0504030811202999],
                                   [0.5023874278457373, -0.1241325147895630,  0.1296836047272858],
                                   [0.5276388478451856,  0.3849458341004425,  0.1860316145060888],
                                   [0.0000000000000000,  0.3211943894582386,  0.2417169802728978],
                                   [0.4723611521548139, -0.1426930137447434,  0.2974023460397066],
                                   [0.4976125721542629,  0.3734800573646999,  0.3537503558185172],
                                   [-0.1826456224014149,  0.4911481706054951,  0.4330308794254957],
                                   [0.1623556467656074,  0.1719735245656081,  0.4333352174509432],
                                   [0.9042910212375085,  0.3461226347221488,  0.0332228202066092],
                                   [0.1223764358558819,  0.9554724347126021,  0.0372674599387856],
                                   [0.5117667223244804,  0.7411556502875718,  0.0303113703466690],
                                   [0.8071820767820592,  0.0032450800326675,  0.1000165757662356],
                                   [0.2018267610657886,  0.6561889120119812,  0.1005751460972309],
                                   [0.4459492087822584,  0.1493366304837961,  0.1475546724845454],
                                   [0.5578864433628844,  0.7055408380929439,  0.1684513999318284],
                                   [0.8495841922145082,  0.4423847101245219,  0.2069432319209446],
                                   [0.2553106121767672,  0.2489757872492882,  0.2207069021286533],
                                   [0.7446893878232328, -0.0063348249274718,  0.2627270584171426],
                                   [0.1504158077854917,  0.5928005179100074,  0.2764907286248505],
                                   [0.4421135566371154,  0.1476543947300661,  0.3149825606139671],
                                   [0.5540507912177420,  0.7033874217015301,  0.3358792880612568],
                                   [0.7981732389342110,  0.4543621509461991,  0.3828588144485717],
                                   [0.1928179232179409,  0.1960630032506014,  0.3834173847795598],
                                   [0.4882332776755194,  0.2293889279630979,  0.4531225901991338],
                                   [-0.1223764358558823,  0.8330959988567119,  0.4461665006070104],
                                   [0.0957089787624908,  0.4418316134846470,  0.4502111403391866]],
                        Si     = 4,
                        O = 6# Pseudpotential valence charge
                    )

                    tiled = system.structure.tile([[a, b, 0], [c, d, 0], [0, 0, 1]])
                    tiled.add_symmetrized_kmesh(kgrid=(1,1,1),kshift=(0,0,0))
                    supercell = generate_physical_system(structure = tiled)
                    #print(type(supercell))
                    #print(supercell.structure.axes)
                    #print(supercell.structure.axes[0][0],supercell.structure.axes[0][1])
                    #print(supercell.structure.axes[1][0],supercell.structure.axes[1][1])
                    a1 = np.sqrt(supercell.structure.axes[0][0]**2 + supercell.structure.axes[0][1]**2)
                    a2 = np.sqrt(supercell.structure.axes[1][0]**2 + supercell.structure.axes[1][1]**2)
                    area = np.abs(supercell.structure.axes[0][0]*supercell.structure.axes[1][1] - supercell.structure.axes[0][1]*supercell.structure.axes[1][0])
                    shape1.append([a1, a2, area/(a1*a2), a, b, c, d])
                    #scf = generate_poscar(structure=tiled, coord='direct')
                    #scf.write('POSCAR-'+str(i+1)+'-SiO2.vasp')
                    system1 = generate_physical_system(
                        units='A',  # Angstrom units
                        axes=[[3.19736567603992, 0, 0],  # Cell axes
                              [0, 4.5823724874315861, 0],
                              [0, 0, 20]],
                        elem=['P', 'P', 'P', 'P'],  # Element names
                        posu=[[0.0000000772737525, 0.4093164276512488, 0.6026039683332174],
                              [-0.0000000223343541, 0.5906834575994857, 0.4973960230667684],
                              [0.4999998793729455, 0.0906834481199840, 0.6026039464980237],
                              [0.5000000656876561, 0.9093166666292746, 0.4973960621019864]],
                        P=5,  # Pseudpotential valence charge
                    )

                    #print(a, b, c, d)
                    tiled1 = system1.structure.tile([[a, b, 0], [c, d, 0], [0, 0, 1]])
                    tiled1.add_symmetrized_kmesh(kgrid=(1, 1, 1), kshift=(0, 0, 0))
                    supercell1 = generate_physical_system(structure=tiled1, P=5)
                    #print(supercell1.structure.axes[0][0],supercell1.structure.axes[0][1])
                    #print(supercell1.structure.axes[1][0],supercell1.structure.axes[1][1])
                    a3 = np.sqrt(supercell1.structure.axes[0][0] ** 2 + supercell1.structure.axes[0][1] ** 2)
                    a4 = np.sqrt(supercell1.structure.axes[1][0] ** 2 + supercell1.structure.axes[1][1] ** 2)
                    area1 = np.abs(
                        supercell1.structure.axes[0][0] * supercell1.structure.axes[1][1] - supercell1.structure.axes[0][
                            1] * supercell1.structure.axes[1][0])
                    shape2.append([a3, a4, area1 / (a3 * a4), a, b, c, d])
                    #scf1 = generate_poscar(structure=tiled, coord='direct')
                    #scf1.write('POSCAR-' + str(i + 1) + '-P.vasp')
                    i1 = i1 + 1

shape1_f = np.array(shape1) #SiO2
shape2_f = np.array(shape2) #P
up_boundary = 0.1
up_boundary_angle = 0.001
down_boundary = -0.1
old_a = 3.19736567603992
old_b = 4.5823724874315861
for i in range(len(shape2_f)):
    for j in range(len(shape1_f)):
        x1 = int(shape2_f[i][3])
        y1 = int(shape2_f[i][4])
        x2 = int(shape2_f[i][5])
        y2 = int(shape2_f[i][6])
        #print(x1, y1, x2, y2)
        new_a = float(shape1_f[j][0])
        new_b = float(shape1_f[j][1])
        #print(new_a, new_b)
        if (x2 * x2 * y1 * y1 - y2 * y2 * x1 * x1) != 0:
            new_a_1 = np.sqrt(np.abs((y1 * y1 * new_b * new_b - y2 * y2 * new_a * new_a) / (x2 * x2 * y1 * y1 - y2 * y2 * x1 * x1)))
            new_b_1 = np.sqrt(np.abs((x1 * x1 * new_b * new_b - new_a * new_a * x2 * x2) / (x1 * x1 * y2 * y2 - x2 * x2 * y1 * y1)))
            strain_a = (float(new_a_1) - old_a) / old_a
            strain_b = (float(new_b_1) - old_b) / old_b
            #print(strain_a, strain_b)
            if down_boundary < strain_a <= up_boundary and down_boundary < strain_b <= up_boundary and down_boundary < abs((shape1_f[j][2] - shape2_f[i][2])/shape1_f[j][2]) <= up_boundary_angle:
                print('OK')
                print(shape1_f[j][0], shape1_f[j][1], shape1_f[j][2])
                print(shape2_f[i][0], shape2_f[i][1], shape2_f[i][2])
                output.writelines('SiO2' + ' ' + str(shape1_f[j][0])+' '+ str(shape1_f[j][1])+' '+ str(shape1_f[j][2])+' '+ str(shape1_f[j][3])+' '+ str(shape1_f[j][4])+' '+ str(shape1_f[j][5])+' '+ str(shape1_f[j][6])+' ' \
                               + 'P'+' ' +str(shape2_f[i][0])+' '+ str(shape2_f[i][1])+' '+ str(shape2_f[i][2])+' '+ str(shape2_f[i][3])+' '+ str(shape2_f[i][4])+' ' + str(shape2_f[i][5])+' '+ str(shape2_f[i][6]) + ' ' +
                                  str(new_a_1) + ' ' + str(new_b_1) + ' ' + str(strain_a) + ' ' + str(strain_b) + ' ' +'\n')

print('done')