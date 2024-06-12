import numpy as np

def create_Ybr(linedata, shunt_susceptances):
    nl = linedata[:,0].astype(int)
    nr = linedata[:,1].astype(int)
    R = linedata[:,2]
    X = linedata[:,3]
    a = linedata[:,8] 

    for i, item in enumerate(a):
        if item == 0:
            a[i] = 1

    num_lines = nl.size
    num_buses = int(max(np.max(nl), np.max(nr)))
    Z = R + 1j * X 
    y = np.ones(num_lines) / (a * Z)

    Ybr = np.zeros((num_buses, num_buses), dtype=complex)
    Ysh = np.zeros((num_buses, num_buses), dtype=complex)

    for j in range(num_lines):
        Ybr[(nl[j]-1,nr[j]-1)] = y[j]
        Ybr[(nr[j]-1,nl[j]-1)] = y[j]

    for k in range(num_lines):
        i1 = int(linedata[k, 0]) - 1 
        i2 = int(linedata[k, 1]) - 1
        Ysh[i1, i1] +=  1j * linedata[k, 4]/2 + y[k] * (1 / (a[k])**2 - 1 / a[k])
        Ysh[i2, i2] +=  1j * linedata[k, 4]/2 + y[k] * (1 - 1 / a[k])
        
    # Assuming shunt_susceptances is a 1D array
    shunt_susceptances = np.array(shunt_susceptances).flatten()
    Ysh += 1j * np.diag(shunt_susceptances)

    return Ybr, Ysh


def create_lookup_matrix(asset_to_bus, num_bus):
    # Create a matrix with zeros
    lookup_matrix = np.zeros((len(asset_to_bus), num_bus), dtype=int)

    # Fill the matrix with 1 where there's a link between the asset and the bus
    for i in range(len(asset_to_bus)):
        lookup_matrix[i, asset_to_bus[i] - 1] = 1  # Adjust for 0-based indexing in Python

    return lookup_matrix