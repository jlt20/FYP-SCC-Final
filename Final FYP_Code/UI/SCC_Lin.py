import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import math

def SCC_linearise(I_sc, I_sc_lim, m, x_wind, N, n_genBus, num_bus, dispatchable):
    xg = np.zeros((N-1,n_genBus))
    if n_genBus == 1:
        unique_int = 1
    else: 
        unique_int = math.comb(n_genBus,2) + n_genBus

    # Represents interaction between 2 unique SGs 
    eta = np.zeros((N-1,unique_int))

    for i_sen in range(1,N):
        mix = format(i_sen, '010b') #Start at 0000000001 to 1111111111
        for imt in range(n_genBus):
            xg[i_sen-1][imt] = int(mix[imt])
        ii = 0
        for ig in range(n_genBus):
            for igg in range(ig, n_genBus):
                ii += 1
                eta[i_sen-1][ii-1] = xg[i_sen-1][ig] * xg[i_sen-1][igg]    
    pass

    x_g_wind = np.tile(xg, (n_genBus,1))
    eta_wind = np.tile(eta, (n_genBus,1))

    x_wind_column = x_wind

    # Create a column of ones with the appropriate shape
    ones_column = np.ones((x_g_wind.shape[0], 1))

    # Concatenate all the columns to form matrix X
    if dispatchable == False:
        X = np.concatenate((ones_column, x_g_wind, eta_wind, x_wind_column), axis=1)
    else:
        X =  np.concatenate((ones_column, x_g_wind, x_wind_column), axis=1)

    b_opt = np.zeros((X.shape[1],num_bus))

    for jj in range(num_bus):
        model = gp.Model()
        model.setParam('IntFeasTol',1e-06)
        model.setParam('ScaleFlag',1)
        model.setParam('BarIterLimit',2000000000)
        model.setParam('CrossoverBasis',0)
        model.setParam('MinRelNodes',0)
        model.setParam('NodeMethod',1)
        model.setParam('PumpPasses',0)
        model.setParam('ZeroObjNodes',0)
        model.setParam('AggFill',10)
        model.setParam('PreSparsify',0)
        model.setParam('TuneTimeLimit',100)
        model.setParam('TuneTrials',2)

        # Define decision variables
        DV_b1 = model.addVars(len(X[0]), vtype=GRB.CONTINUOUS, name="DV_b1", lb = -GRB.INFINITY, ub = GRB.INFINITY)
        
        # Set objective function
        objective = 0
        for n in range(len(I_sc)):
            # If SCC < Limit, add constraint
            if I_sc[n][jj] <= I_sc_lim:
                model.addConstr(gp.quicksum(X[n, i] * DV_b1[i] for i in range(len(DV_b1))) <= I_sc_lim - 0.001)
            else:
                # If SCC > Limit + offset, add constraint
                if I_sc[n][jj] > I_sc_lim + m:
                    model.addConstr(gp.quicksum(X[n, i] * DV_b1[i] for i in range(len(DV_b1))) >= I_sc_lim + 0.001)
                # Minimise sum of least squares of calculated SCC vs linearised SCC
                else:
                    objective = objective + ((I_sc[n][jj] - gp.quicksum(X[n, i] * DV_b1[i] for i in range(len(DV_b1)))) ** 2) / (I_sc[n][jj])**2
            
        # Additional constraints
        for i in range(len(DV_b1)):
            model.addConstr(DV_b1[i] <= 200)
        for i in range(len(DV_b1)):
            model.addConstr(-DV_b1[i] <= 200)
            
        # Set objective
        model.setObjective(objective, GRB.MINIMIZE)
        
        # Optimize the model
        model.setParam('OutputFlag', 1)
        model.optimize()
        for i in range(len(DV_b1)):
            b_opt[i][jj] = DV_b1[i].X

    I_sc_linearised = X @ b_opt

    return I_sc_linearised, b_opt