import numpy as np
import pandapower as pp
import pandas as pd
from pandapower.networks import create_cigre_network_mv
import copy

def init_cigre():
    # Set NumPy print options for better readability (optional)
    np.set_printoptions(precision=4, suppress=True, linewidth=150)

    # Creates MV CIGRE Network
    net = create_cigre_network_mv()
    # Max no of columns enabled
    pd.set_option('display.max_columns', None)

    # test generator.
    # pp.create_gen(net, bus=5, p_mw=1.0, vm_pu=1.0)

    # run pp
    pp.runpp(net)
    print("\n--- CONTENTS OF NET.RES_BUS AFTER PANDAPOWER RUN ---")
    print(net.res_bus)  # This will print the entire result table
    print("-------------------------------------------------------")

    return net
