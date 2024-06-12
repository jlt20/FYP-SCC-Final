import numpy as np
import module
import case39_ibr # Change this to any other line data file

def network_Param(gen, load, wind):
    ppc = case39_ibr.case_run()
    linedata_array = ppc["branch"]
    num_bus = np.max(linedata_array[:,0:2].astype(int))
    shunt_susceptances = np.zeros((num_bus, 1))

    # Add shunt susceptance values to the corresponding bus indices
    for line in linedata_array:
        bus_nl = int(line[0]) - 1
        bus_nr = int(line[1]) - 1
        Bsh_nl = line[4]/2
        Bsh_nr = line[4]/2
        shunt_susceptances[bus_nl] += Bsh_nl
        shunt_susceptances[bus_nr] += Bsh_nr
    
    Ybr, Ysh = module.create_Ybr(linedata_array, shunt_susceptances)

    Gbr = np.real(Ybr)
    Bbr = np.imag(Ybr)
    Gsh = np.real(Ysh)
    Bsh = np.imag(Ysh)
    Vmin = np.ones((num_bus,1)) * 0.9
    Vmax = np.ones((num_bus,1)) * 1.1
    Pijmax = np.ones((num_bus,num_bus)) * 1000
    S_limit = np.ones((num_bus,num_bus)) * 0.1

    net_gen = np.array(gen)
    net_load = np.array(load)
    net_wind = np.array(wind)
    net_gen_mat = module.create_lookup_matrix(net_gen, num_bus)
    net_load_mat = module.create_lookup_matrix(net_load, num_bus)
    net_wind_mat = module.create_lookup_matrix(net_wind, num_bus)
    
    return Gbr, Bbr, Gsh, Bsh, num_bus, Vmin, Vmax, Pijmax, S_limit, net_gen, net_gen_mat, net_load_mat, net_wind_mat, linedata_array, ppc

