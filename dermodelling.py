import pandapower as pp
import copy
import dlpfcoeff
import solvedlpf
import erroranalysisdlpf
# handles addition of gen or sgen according to the config dictionary and then runs a pp analysis.
def der_modelling(net, der_config, A, B, C, D, pq_bus, pv_bus, sl_bus, num_total_buses):
    #copies base net to test components with this one.
    copied_net = copy.deepcopy(net)

    #{type: 'sgen' or 'gen', bus, p_mw, vm_pu } - structure of der_config; using for loop for adding multiple gen/sgen functionality

    if der_config['type'] == 'sgen':
        pp.create_sgen(copied_net,der_config['bus'], der_config['p_mw'], der_config['q_mvar'])
        relinearize = False

    elif der_config['type'] == 'gen':
        pp.create_gen(copied_net,der_config['bus'], der_config['p_mw'], der_config['vm_pu'])
        relinearize = True

    #run pp
    pp.runpp(copied_net)

    if relinearize:
        A_new, B_new, C_new, D_new, pv_bus_new, pq_bus_new, sl_bus_new, num_total_buses_new = dlpfcoeff.get_dlpf_coeff(copied_net)
        dlpf_vm_pu, dlpf_theta_rad = solvedlpf.solve_dlpf(copied_net, A_new, B_new, C_new, D_new, pq_bus_new, pv_bus_new,sl_bus_new, num_total_buses_new)
    elif relinearize == False:
        dlpf_vm_pu, dlpf_theta_rad = solvedlpf.solve_dlpf(copied_net, A, B, C, D, pq_bus, pv_bus, sl_bus, num_total_buses)

    erroranalysisdlpf.error_analysis_dlpf(copied_net, dlpf_vm_pu, dlpf_theta_rad)
