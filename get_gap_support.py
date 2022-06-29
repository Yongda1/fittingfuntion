import numpy as np
def get_gap(par1, par2, par3):
    gs_file = open(par1, "r")
    exc_file = open(par2, "r")
    output = open(par3, "w")
    temp_gs = []
    temp_exc = []
    for line in gs_file.readlines():
        temp_gs.append(line.split())
    for line in exc_file.readlines():
        temp_exc.append(line.split())

    row_number = len(temp_gs)
    temp1 = np.zeros((row_number, 4))
    for i in range(row_number):
        n1 = 0
        for j in [0, 1, 6, 7]:
            temp1[i][n1] = temp_gs[i][j]
            n1 = n1 + 1
    # print(temp1)
    print('------------------------------------------------')
    temp2 = np.zeros((row_number, 4))
    for i in range(row_number):
        n1 = 0
        for j in [0, 1, 6, 7]:
            temp2[i][n1] = temp_exc[i][j]
            #print(i)
            #print(j)
            #print(n1)
            n1 = n1 + 1
    # print(temp2)
    temp_f_1 = []
    data_number = 100
    for i in range(len(temp1)):
        mu_gs = temp1[i][2]
        sigma_gs = temp1[i][3]
        data_array_gs = np.random.normal(mu_gs, sigma_gs, data_number)
        mu_exc = temp2[i][2]
        sigma_exc = temp2[i][3]
        data_array_exc = np.random.normal(mu_exc, sigma_exc, data_number)
        gap = []
        for a in range(len(data_array_exc)):
            for b in range(len(data_array_gs)):
                gap.append(data_array_exc[a] - data_array_gs[b])
        gap_f = np.array(gap)
        energy = gap_f.mean()
        error = gap_f.std()
        # print(temp1[i][0], temp1[i][1], energy, error)
        temp_f_1.append([temp1[i][0], temp1[i][1], energy, error])
        output.writelines(str(temp1[i][0]) + ' ' + str(temp1[i][1]) + ' ' + str(energy) + ' ' + str(error) + '\n')

    return temp_f_1

def get_gs(par1, par2, par3):
    gs_file = open(par1, "r")
    exc_file = open(par2, "r")
    output = open(par3, "w")
    temp_gs = []
    temp_exc = []
    for line in gs_file.readlines():
        temp_gs.append(line.split())
    for line in exc_file.readlines():
        temp_exc.append(line.split())

    row_number = len(temp_gs)
    temp1 = np.zeros((row_number, 4))
    for i in range(row_number):
        n1 = 0
        for j in [0, 1, 6, 7]:
            temp1[i][n1] = temp_gs[i][j]
            n1 = n1 + 1
    # print(temp1)
    print('------------------------------------------------')
    temp2 = np.zeros((row_number, 4))
    for i in range(row_number):
        n1 = 0
        for j in [0, 1, 6, 7]:
            temp2[i][n1] = temp_exc[i][j]
            n1 = n1 + 1
    # print(temp2)
    temp_f_1 = []
    data_number = 100
    for i in range(len(temp1)):
        mu_gs = temp1[i][2]
        sigma_gs = temp1[i][3]
        data_array_gs = np.random.normal(mu_gs, sigma_gs, data_number)
        mu_exc = temp2[i][2]
        sigma_exc = temp2[i][3]
        data_array_exc = np.random.normal(mu_exc, sigma_exc, data_number)
        energy_gs = data_array_gs.mean()
        error_gs = data_array_gs.std()
        # print(temp1[i][0], temp1[i][1], energy, error)
        temp_f_1.append([temp1[i][0], temp1[i][1], energy_gs, error_gs])
        output.writelines(str(temp1[i][0]) + ' ' + str(temp1[i][1]) + ' ' + str(energy_gs) + ' ' + str(error_gs) + '\n')

    return temp_f_1