import numpy as np
import pandas as pd
import copy

def system_Param():
    FreC = 0
    days = 1
    stages = 24
    df1 = pd.read_excel(r"C:\Users\Ashley\OneDrive - MMU\Desktop\Lun\FYP\FYP_Code\Data\Demand_39_node_MILP_varying.xlsx", sheet_name='Demand_active')
    df2 = pd.read_excel(r"C:\Users\Ashley\OneDrive - MMU\Desktop\Lun\FYP\FYP_Code\Data\Demand_39_node_MILP_varying.xlsx", sheet_name='Demand_reactive')
    df3 = pd.read_excel(r"C:\Users\Ashley\OneDrive - MMU\Desktop\Lun\FYP\FYP_Code\Data\ScenarioTree_39_node_MILP_wind_1st.xlsx", sheet_name='Scenario_tree_active', header=None)
    df4 = pd.read_excel(r"C:\Users\Ashley\OneDrive - MMU\Desktop\Lun\FYP\FYP_Code\Data\ScenarioTree_39_node_MILP_wind_1st.xlsx", sheet_name='Scenario_tree_reactive', header=None)

    # Extract the desired subset of data
    # subset_active = df1.iloc[1:25, :19]
    subset_active = df1.iloc[:, :-2]

    # Find the first row with all NaN values
    first_empty_row_index = df2.index[df2.isnull().all(axis=1)][0]
    subset_reactive = df2.iloc[:first_empty_row_index, :-3]

    # Convert the Pandas DataFrame to a NumPy array and transpose it
    Demand_active = subset_active.to_numpy().T
    Demand_reactive = subset_reactive.to_numpy().T

    # Print or use the resulting NumPy array as needed
    Demand_active = 1e-2 * Demand_active
    Demand_reactive = 1e-2 * Demand_reactive

    num_load = Demand_active.shape[0]

    Scenario_tree_active = df3.to_numpy().T
    Scenario_tree_reactive = df4.to_numpy().T
    Wind_scenarios_active = 1e-2 * Scenario_tree_active
    Wind_scenarios_reactive = 1e-2 * Scenario_tree_reactive

    print(len(Scenario_tree_active))

    Wind_scenarios_active = np.reshape(Wind_scenarios_active, (1, 4, 24))
    Wind_scenarios_reactive = np.reshape(Wind_scenarios_reactive, (1, 4, 24))

    num_wind = Wind_scenarios_active.shape[1]

    Demand_active_day_matrix = Demand_active.copy()
    Demand_reactive_day_matrix = Demand_reactive.copy()
    Wind_scenarios_day_active_matrix = Wind_scenarios_active.copy()
    Wind_scenarios_day_reactive_matrix = Wind_scenarios_reactive.copy()

    Demand_active_day_cell = np.empty((1, 24), dtype=object)
    Demand_reactive_day_cell = np.empty((1, 24), dtype=object)
    Wind_scenarios_day_active_cell = np.empty((1, 24), dtype=object)
    Wind_scenarios_day_reactive_cell = np.empty((1, 24), dtype=object)

    for j in range(stages):
        Demand_active_day_cell[0, j] = Demand_active_day_matrix[:,j]
        Demand_reactive_day_cell[0, j] = Demand_reactive_day_matrix[:,j]
        Wind_scenarios_day_active_cell[0, j] = Wind_scenarios_active[:,:,j]
        Wind_scenarios_day_reactive_cell[0, j] = Wind_scenarios_reactive[:,:,j]

    Demand_day_active = np.empty((1, 1), dtype=object)
    Demand_day_reactive = np.empty((1, 1), dtype=object)
    Wind_scenarios_day_active = np.empty((1, 1), dtype=object)
    Wind_scenarios_day_reactive = np.empty((1, 1), dtype=object)

    Demand_day_active[0,0] = Demand_active_day_cell
    Demand_day_reactive[0,0] = Demand_reactive_day_cell
    Wind_scenarios_day_active[0,0] = Wind_scenarios_day_active_cell
    Wind_scenarios_day_reactive[0,0] = Wind_scenarios_day_reactive_cell

    scenarios = 1
    prob_tree = 1/scenarios*np.ones((scenarios,stages))

    factor = 1

    NLHR = np.array([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]) #k£/h fixed generation cost\
    HRS = np.array([17, 27, 37, 47, 57, 67, 77, 87, 97, 107]) * 0.1 # variable generation cost k£/100MWh
    stc = np.array([0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]) * 1 * factor # k£ Start up cost

    num_gen = len(HRS)

    array1 = np.array([1100, 646, 725, 652, 508, 687, 580, 564, 865, 1040]) / 100
    array1 = np.reshape(array1, (10,1))

    Gen_limits_active = np.zeros((10, 1))
    Gen_limits_active = np.concatenate([Gen_limits_active, array1], axis=1)

    array2 = np.ones((10,1)) * 100
    Gen_limits_reactive = -np.ones((10, 1)) * 100
    Gen_limits_reactive = np.concatenate([Gen_limits_reactive, array2], axis=1)

    Ramp_limits = np.ones((10,2)) * 30
    Updowntime_limits = np.zeros((10,2))

    if Gen_limits_active.shape[0] != num_gen:
        raise ValueError('Matrices "Gen_limits_active" and "Ramp_limits" must have as many rows as generators defined')
    if Gen_limits_active.shape[1] != 2:
        raise ValueError('Matrices "Gen_limits_active" and "Ramp_limits" must only have 2 columns')
    if not np.array_equal(np.shape(Gen_limits_active), np.shape(Ramp_limits)):
        raise ValueError('There is a problem with the size of matrix "Gen_limits_active" or "Ramp_limits"')
    if not np.array_equal(np.shape(Gen_limits_active), np.shape(Gen_limits_reactive)):
        raise ValueError('There is a problem with the size of matrix "Gen_limits_active" or "Gen_limits_reactive"')

    return days, Demand_active, Demand_day_active, Demand_day_reactive, Demand_reactive, factor, FreC, Gen_limits_active, Gen_limits_reactive, HRS, NLHR, num_gen, num_load, num_wind, prob_tree, Ramp_limits, Scenario_tree_active, Scenario_tree_reactive, scenarios, stages, stc, Updowntime_limits, Wind_scenarios_active, Wind_scenarios_day_active, Wind_scenarios_day_reactive, Wind_scenarios_reactive

system_Param()