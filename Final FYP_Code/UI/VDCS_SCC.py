from pypower.api import runpf, case39
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import scipy.io
import math

def export_to_excel(array, file_name='output.xlsx', sheet_name='Sheet1'):
    """
    Export a numpy array to an Excel sheet.

    Parameters:
    - array: numpy array to be exported
    - file_name: name of the Excel file (default: 'output.xlsx')
    - sheet_name: name of the sheet in the Excel file (default: 'Sheet1')
    """
    # Convert the numpy array to a pandas DataFrame
    df = pd.DataFrame(array)

    # Write the DataFrame to an Excel file
    df.to_excel(file_name, sheet_name=sheet_name, index=False)

def adjust_generators(ppc):
    total_load = np.sum(ppc['bus'][:, 2]) # Sum of all loads (Pd)
    total_capacity = np.sum(ppc['gen'][:, 8]) # Sum of generator capacity (Pmax)

    # Calculate new load for each generator based on its capacity
    for i in range(len(ppc['gen'])):
        capacity = ppc['gen'][i, 8] # Pmax of the generator
        ppc['gen'][i, 1] = (capacity / total_capacity) * total_load # New load for the generator

    return ppc

def cal_Y39(input_ppc, gen_loc, gen_value, num_lines, linedata, num_bus):
    ppc = input_ppc
    ppc = adjust_generators(ppc) # Comment this out depending on wheter Pg and Qg are filled in. 
    results, success = runpf(ppc)

    # Extract the magnitude and voltage at each bus
    bus_data = results['bus']
    bus_magnitudes = bus_data[:, 7]  # Voltage magnitudes (Vm)
    bus_angles = bus_data[:, 8]      # Voltage angles (Va)

    # Extract the active (P) and reactive (Q) power for loads
    bus_p_load = bus_data[:, 2]  # P (MW) load
    bus_q_load = bus_data[:, 3]  # Q (MVAr) load

    # Convert angles from degrees to radians
    bus_angles_rad = np.radians(bus_angles)

    # Compute the complex voltages
    complex_voltages = bus_magnitudes * (np.cos(bus_angles_rad) + 1j * np.sin(bus_angles_rad))
    base_power = 100.0
    s_load = (bus_p_load + 1j * bus_q_load)/base_power

    # Compute the load currents
    I_L = np.zeros(len(complex_voltages), dtype=complex) # Double Checked calculations

    for i in range(len(complex_voltages)):
        s = s_load[i]
        v = complex_voltages[i]
        if v != 0:
            I_L[i] = np.conj(s / v)
        else:
            I_L[i] = 0

    # No.of generators in system
    num_gen = len(gen_loc)

    # Line Admittance Matrix: Y0
    # Takes reactance from linedata to get admittance (inverse)
    Line_39 = np.zeros((num_lines, 4))
    # Should be getting from ppc instead of linedata, to change
    for i in range(num_lines):
        Line_39[i][0] = linedata[i][0]
        Line_39[i][1] = linedata[i][1]
        Line_39[i][2] = linedata[i][2]
        Line_39[i][3] = linedata[i][3]

    Y39 = np.zeros((num_bus, num_bus), dtype=complex) 

    for k in range(len(Line_39)):
        idx0 = int(Line_39[k][0]) - 1 
        idx1 = int(Line_39[k][1]) - 1
        R = Line_39[k][2]
        X = Line_39[k][3]
        const = 1/(R - 1j*X)
        # The off-diagonal elements Y39[i][j] and Y39[j][i] are decremented by the line admittance. 
        # Connection between buses i and j through a line with certain admittance, indicating that power can flow between these buses.
        Y39[idx0][idx1] = Y39[idx0][idx1] - const
        Y39[idx1][idx0] = Y39[idx1][idx0] - const
        # The diagonal elements Y39[i][i] and Y39[j][j] are incremented by the line admittance. 
        # This reflects the sum of all power entering or leaving a bus must equal zero. 
        # Adding the line admittance to the diagonal entries effectively accounts for the power leaving the bus via this line.
        Y39[idx0][idx0] = Y39[idx0][idx0] + const
        Y39[idx1][idx1] = Y39[idx1][idx1] + const

    gen_39 = np.zeros((num_gen, 3), dtype=complex)
    for i in range(num_gen):
        gen_39[i][0] = int(gen_loc[i])
        gen_39[i][1] = 1
        # Generator Admittance
        gen_39[i][2] = 1j * (0.2 * 100 /gen_value[i]) #This is X''dg in p.u

    return Y39, gen_39, complex_voltages, I_L

def cal_Z_LUT(num_gen, num_bus, Y39, gen_39):
    N = 2**(num_gen)
    n_genBus = num_gen
    Z_LUT = np.zeros((N-1, num_bus**2), dtype=complex)
    x_g = np.zeros((n_genBus,1))

    #Y_new = Y0 + Yg, add in admittance contribution from generator
    for i in range(1, N):
        Y_new = np.array(Y39)
        # Go through generator scenarios from 0000000001 to 1111111111
        mix = format(i, '010b')
        for j in range(n_genBus):
            #x_g = Whether generator is on or off, based on iteration
            x_g[j] = float(mix[j])
            if x_g[j] == 0:
                pass
            else:
                idx = int(gen_39[j][0]) - 1  
                const = gen_39[j][2]  
                Y_new[idx][idx] = Y39[idx][idx] + (1/(-const) * x_g[j])

        Z39 = np.linalg.inv(Y_new) 
        for m in range(num_bus):
            # Z39 is 39x39, Z_LUT essentially flattens that to represent one specific scenario
            Z_LUT[i-1, m*num_bus:(m+1)*num_bus] = Z39[m, :num_bus]

    return Z_LUT

def cal_VDCS_SCC(N_wind, i_GL, num_gen, num_bus, complex_voltages, Z_LUT, I_L, ppc):
    extracted_values = []
    for node in i_GL:
        # Find the row where the first column matches the node
        row = ppc['gen'][ppc['gen'][:, 0] == node]
        if row.size > 0:  # Check if a matching row is found
            extracted_values.append(row[0, 8])
    wind_power = np.array(extracted_values)
    dc = wind_power/100

    N = 2 **(num_gen)
    x_gen_wind = np.zeros(((N-1)*N_wind, 1))
    Isc = np.zeros(((N-1)*N_wind,num_bus), dtype=complex)
    dV_c_i = np.zeros(((N-1)*N_wind,num_bus*len(i_GL)), dtype=complex)
    sum_c = np.zeros((num_bus, 1), dtype=complex)
    sum_ibr = np.zeros((num_bus, len(i_GL)), dtype=complex)
    I_sc = np.zeros((num_bus, 10000), dtype=complex)
    Z_new = np.zeros((num_bus, num_bus), dtype=complex)
    Ifc = np.zeros((num_bus, 1), dtype=complex)
    dV_c = -1 * np.ones((num_bus, len(i_GL)), dtype=complex)
    dV_c_pre = -1 * np.ones((num_bus, len(i_GL)), dtype=complex)
    for i_w in range(N_wind):
        alpha = (1/N_wind)*(i_w+1)
        for i in range(1,N):
            
            for x in range(num_bus):
                Z_new[x][:] = Z_LUT[(i - 1), num_bus * x:num_bus * (x+ 1)]

            for F in range(num_bus): #num_bus
                I_sc[F, :] = 0
                Ifc[F, :] = 0
                # dV_c[F][len(i_GL)-1] = -1 # Not sure the point of this

                for ii in range(1,10000):
                    sum_c[F] = 0
                    sum_ibr[F] = 0
                    dV_c_pre = dV_c.copy()
                    dV_c[F] = -1

                    for m in range(len(i_GL)):
                        if ii == 1:   
                            Ir = -1j * dV_c[F][m] * alpha * dc[m] # 1j is correct
                            Ifc[i_GL[m]-1] = Ir
                        else:
                            V_c = dV_c[F][m] + complex_voltages[i_GL[m]-1] # Lies between 0 and 1, p.u.
                            dmagV_c = np.abs(V_c) - np.abs(complex_voltages[i_GL[m]-1])
                            temp_Ir = max(-1.2 * dc[m], dmagV_c * dc[m] * alpha)
                            temp_Ia = np.real(I_L[i_GL[m] - 1])
                            if temp_Ia > np.sqrt((1.2*dc[m])**2 - temp_Ir**2):
                                temp_Ia = np.sqrt((1.2*dc[m])**2 - temp_Ir**2)
                                print(alpha)
                            temp_Ifc = temp_Ia - 1j * temp_Ir
                            Ifc[i_GL[m]-1] = temp_Ifc

                        sum_c[F] = sum_c[F] + Z_new[F][i_GL[m]-1] *(Ifc[i_GL[m]-1] - I_L[i_GL[m]-1])
                    I_sc[F][ii] = -(complex_voltages[F]+sum_c[F]) / Z_new[F][F]

                    for c in range(len(i_GL)):
                        for c_apo in range(len(i_GL)):
                            sum_ibr[F][c] += Z_new[i_GL[c_apo]-1][i_GL[c]-1] * (Ifc[i_GL[c_apo]-1] - I_L[i_GL[c]-1])
                        dV_c[F][c] = sum_ibr[F][c] + I_sc[F][ii] * Z_new[F][i_GL[c]-1]
                        
                    # print(dV_c)
                    if (np.abs(I_sc[F][ii] - I_sc[F][ii-1])  < 1e-6 and np.abs(np.mean(dV_c[F,:] - dV_c_pre[F,:])) < 1e-3):
                        break

                Isc[(N-1)*(i_w)+(i-1)][F] = I_sc[F][ii]

                dV_c_i[(N-1)*(i_w)+(i-1)][F*len(i_GL):F*len(i_GL)+len(i_GL)] = dV_c[F,:]
            x_gen_wind[(N-1)*(i_w)+(i-1)][-1] = alpha
    
    export_to_excel(np.abs(Isc), "Isc_new.xlsx")
    #export_to_excel(x_gen_wind, "x_gen_wind.xlsx") # Need to fix this
    return np.abs(Isc), x_gen_wind