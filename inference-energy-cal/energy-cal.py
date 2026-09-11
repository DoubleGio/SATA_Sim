import yaml
import os
import math
import argparse
from conv_utils import compute_conv_output_size

def read_cycles(cycle_filepath, arch_configpath, workload_filepath):

    with open(arch_configpath, 'r') as file:
        data = yaml.safe_load(file)
    arch = data.get('architecture')
    dataflow = arch.get('dataflow')
    with open(workload_filepath, 'r') as file:
        work_data = yaml.safe_load(file)
    general = work_data.get('General')
    timestep = general.get('timestep')

    with open(cycle_filepath, 'r') as file:
        data = yaml.safe_load(file)
    
    cycle_stats = {}
    cycle_stats['total_cycles'] = 0.0
    cycle_stats['sram_i_reads'] = 0.0
    cycle_stats['sram_w_reads'] = 0.0
    cycle_stats['sram_o_writes'] = 0.0
    cycle_stats['dram_i_reads'] = 0.0
    cycle_stats['dram_w_reads'] = 0.0
    cycle_stats['dram_o_writes'] = 0.0

    if dataflow == 'sata':
        for layer, stats in data.items():
            # print((stats.get('SRAM OFMAP Start Cycle') + stats.get('SRAM OFMAP Cycles'))*timestep)
            cycle_stats['total_cycles'] += (stats.get('SRAM OFMAP Start Cycle') + stats.get('SRAM OFMAP Cycles')) * timestep + stats.get('SRAM Filter Cycles')
            cycle_stats['sram_i_reads'] += stats.get('SRAM IFMAP Reads')
            cycle_stats['sram_w_reads'] += stats.get('SRAM Filter Reads')
            # cycle_stats['sram_o_writes'] += stats.get('SRAM OFMAP Writes') #! This is the arxived version where the writing of psum is not considered.
            cycle_stats['sram_o_writes'] += stats.get('SRAM OFMAP Writes') * timestep
            cycle_stats['dram_i_reads'] += stats.get('DRAM IFMAP Reads')
            cycle_stats['dram_w_reads'] += stats.get('DRAM Filter Reads')
            cycle_stats['dram_o_writes'] += stats.get('DRAM OFMAP Writes')
    else:
        for layer, stats in data.items():
            cycle_stats['total_cycles'] += stats.get('SRAM OFMAP Start Cycle') + stats.get('SRAM OFMAP Cycles')
            cycle_stats['sram_i_reads'] += stats.get('SRAM IFMAP Reads')
            cycle_stats['sram_w_reads'] += stats.get('SRAM Filter Reads')
            cycle_stats['sram_o_writes'] += stats.get('SRAM OFMAP Writes')
            cycle_stats['dram_i_reads'] += stats.get('DRAM IFMAP Reads')
            cycle_stats['dram_w_reads'] += stats.get('DRAM Filter Reads')
            cycle_stats['dram_o_writes'] += stats.get('DRAM OFMAP Writes')
    
    return cycle_stats
    

def extract_workload(workload_filepath):
    """Extract workload information from the workload YAML file."""
    with open(workload_filepath, 'r') as file:
        work_data = yaml.safe_load(file)
    general = work_data.get('General')
    timestep = general.get('timestep')
    sparsity = general.get('sparsity')

    workload_dic = {}
    layers = work_data.get('Layers', {})
    total_mac = 0.0
    neuron_types = {layer['attributes'].get('Neuron Type', 'lif') for layer in layers}
    total_neurons = {n_type: 0.0 for n_type in neuron_types}

    for l in layers:
        name = l['name']
        attr = l['attributes']
        n_type = attr.get('Neuron Type', 'lif')  # Default to 'lif' if not specified
        layer_sparsity = attr.get('sparsity', sparsity)  # per-layer override
        if 'Conv' in name:
            of_h, of_w = compute_conv_output_size(
                attr['IFMAP Height'], attr['IFMAP Width'],
                attr['Filter Height'], attr['Filter Width'],
                attr['Strides'], attr.get('Padding', 'same')
            )
            total_neurons[n_type] += attr['Num Filter'] * (of_w * of_h) * timestep
            total_mac += attr['Num Filter'] * (of_w * of_h) * (attr['Filter Width'] * attr['Filter Height'] * attr['Channels']) * timestep * (1-layer_sparsity)
            if attr.get('Recurrent', False): # Added conv (all-to-all) recurrence (1x1 kernel)
                total_mac += attr['Num Filter'] * attr['Num Filter'] * (of_w * of_h) * timestep * (1-layer_sparsity)
        elif 'FC' in name:
            total_neurons[n_type] += attr['Num Filter'] * timestep
            total_mac += attr['Num Filter'] * attr['Channels'] * timestep * (1-layer_sparsity)
            if attr.get('Recurrent', False): # Added (all-to-all) recurrence
                total_mac += attr['Num Filter'] * attr['Num Filter'] * timestep * (1-layer_sparsity) 
    
    workload_dic['total_mac'] = int(total_mac)
    for n_type, count in total_neurons.items():
        workload_dic[f'total_{n_type}'] = int(count)
    workload_dic['total_neurons'] = int(sum(total_neurons.values()))

    return workload_dic


def comp_computation_energy(comp_filepath, cycle_dict, arch_configpath, workload_dict, res_folder):
    """Compute the energy consumption of processing elements (PEs) based on the workload and architecture configuration."""
    with open(arch_configpath, 'r') as file:
        data = yaml.safe_load(file)
    subtrees_ = data.get('architecture', {}).get('subtree', [])
    for tree in subtrees_:
        if tree.get('class') == 'pe-array':
            attributes = tree.get('attributes')
            height = attributes.get('height')
            width  = attributes.get("width")
    pe_size = height * width
    arch = data.get('architecture')
    freq = arch.get('clock-frequency') * 1000000 #! Need to convert to MHz first, in yaml, M is not take into consideration
    cyc = 1/freq

    with open(comp_filepath, 'r') as file:
        comp_data = yaml.safe_load(file)

    comp_dic = {}
    total_comp = 0.0
    for component, values in comp_data.items():
        if f'total_{component}' in workload_dict:
            workload = workload_dict[f'total_{component}']
        elif component in ['spad', 'spike-mac']:
            workload = workload_dict['total_mac']
        else:
            continue  # Skip components that don't have a corresponding workload
        comp_dic[component] = {}
        subtotal = 0.0
        
        # ! Power of computation unit is in mW, from the comp-stat.yaml
        convert_ratio = 1000000 # ! Need to convert it back to nJ, to align with the memory energy, which is in nJ
        for key, value in values.items():
            if isinstance(value, dict):
                comp_dic[component]['energy-operation'] = value['y'] * workload * cyc * convert_ratio 
                subtotal += comp_dic[component]['energy-operation']
                comp_dic[component]['energy-ungated'] = value['n'] * cycle_dict['total_cycles'] * cyc * convert_ratio * pe_size
                subtotal += comp_dic[component]['energy-ungated']
            elif 'lpower' in key:
                comp_dic[component]['energy-leakage'] = value * cycle_dict['total_cycles'] * cyc * convert_ratio * pe_size
                subtotal += comp_dic[component]['energy-leakage']
        comp_dic[component]['total'] = subtotal
        total_comp += subtotal
    comp_dic['total'] = total_comp

    comp_dic['total_mac_ops'] = workload_dict['total_mac']
    for tag, count in workload_dict.items():
        if tag.startswith('total_') and tag not in ['total_mac', 'total_neurons']:
            comp_dic[f'{tag}_ops'] = count
    comp_dic['total_activation_ops'] = workload_dict['total_neurons']

    file_path = os.path.join(res_folder, 'computation-energy.yaml')
    with open(file_path, 'w') as yaml_file:
        yaml.dump(comp_dic, yaml_file, default_flow_style=False)
    return comp_dic


def comp_mem_energy(mem_filepath, cycle_dict, arch_configpath, res_folder):
    """Compute the energy consumption of memory components (SRAM and DRAM) based on the cycle statistics and architecture configuration."""
    with open(arch_configpath, 'r') as file:
        arch_data = yaml.safe_load(file)
    arch = arch_data.get('architecture')
    freq = arch.get('clock-frequency') * 1000000 #! Need to convert to MHz first, in yaml, M is not take into consideration
    cyc = 1/freq
    x_bw = arch.get('act-prec')
    w_bw = arch.get('weight-prec')
    o_bw = arch.get('output-prec')
    convert_ratio = 1000000 #! Need again to convert SRAM leakage to nJ, the leaking power is in mW from the mem-stats.yaml

    with open(mem_filepath, 'r') as file:
        data = yaml.safe_load(file)
    mem_dic = {}
    dram_entries = data.get('DRAM', [])
    total_dram = 0.0
    for entry in dram_entries:
        name = entry.get('name')
        sub_total = 0.0
        if name not in mem_dic:
            mem_dic[name] = {}
        mem_dic[name]['ifmap'] = entry.get('read energy') * (x_bw/8) * cycle_dict['dram_i_reads'] # ? CHECK: is the energy in nJ ?
        total_dram += mem_dic[name]['ifmap']
        sub_total  += mem_dic[name]['ifmap']
        mem_dic[name]['weight'] = entry.get('read energy') * (w_bw/8) * cycle_dict['dram_w_reads'] # ? CHECK: is the energy in nJ ?
        total_dram += mem_dic[name]['weight']
        sub_total  += mem_dic[name]['weight']
        mem_dic[name]['ofmap'] = entry.get('read energy') * (o_bw/8) * cycle_dict['dram_o_writes'] # ? CHECK: is the energy in nJ ?
        total_dram += mem_dic[name]['ofmap']
        sub_total  += mem_dic[name]['ofmap']
        mem_dic[name]['total'] = sub_total
    mem_dic['dram_total'] = total_dram

    sram_entries = data.get('SRAM', [])
    total_sram = 0.0
    for entry in sram_entries:
        name = entry.get('name')
        if name not in mem_dic:
            mem_dic[name] = {}
        if 'ifmap' in name:
            mem_dic[name]['ifmap-dynamic'] = entry.get('read dynamic energy') * (x_bw/8) * cycle_dict['sram_i_reads'] # ? CHECK: is the energy in nJ ?
            mem_dic[name]['ifmap-leakage'] = entry.get('leakage power') * cycle_dict['total_cycles'] * cyc * convert_ratio #! The power is in mW, need to scale it to nJ
            mem_dic[name]['ifmap-total']= mem_dic[name]['ifmap-dynamic'] + mem_dic[name]['ifmap-leakage']
            total_sram += mem_dic[name]['ifmap-dynamic'] + mem_dic[name]['ifmap-leakage']
        elif 'weight' in name:
            mem_dic[name]['weight-dynamic'] = entry.get('read dynamic energy') * (w_bw/8) * cycle_dict['sram_w_reads'] # ? CHECK: is the energy in nJ ?
            mem_dic[name]['weight-leakage'] = entry.get('leakage power') * cycle_dict['total_cycles'] * cyc * convert_ratio
            mem_dic[name]['weight-total']= mem_dic[name]['weight-dynamic'] + mem_dic[name]['weight-leakage']
            total_sram += mem_dic[name]['weight-dynamic'] + mem_dic[name]['weight-leakage']
        elif 'ofmap' in name:
            mem_dic[name]['ofmap-dynamic'] = entry.get('write dynamic energy') * (o_bw/8) * cycle_dict['sram_o_writes'] # ? CHECK: is the energy in nJ ?
            mem_dic[name]['ofmap-leakage'] = entry.get('leakage power') * cycle_dict['total_cycles'] * cyc * convert_ratio
            mem_dic[name]['ofmap-total']= mem_dic[name]['ofmap-dynamic'] + mem_dic[name]['ofmap-leakage']
            total_sram += mem_dic[name]['ofmap-dynamic'] + mem_dic[name]['ofmap-leakage']
    
    mem_dic['sram_total'] = total_sram
    mem_dic['total'] = total_dram + total_sram

    file_path = os.path.join(res_folder, 'memory-energy.yaml')
    with open(file_path, 'w') as yaml_file:
        yaml.dump(mem_dic, yaml_file, default_flow_style=False)
    return mem_dic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SATA energy calculation.")
    parser.add_argument("-c", "--config", default="archs/sata-config.yaml", help="Path to the configuration YAML file.",)
    parser.add_argument("-w", "--workload", default="workload.yaml", help="Path to the workload YAML file.",)
    args = parser.parse_args()

    res_folder = os.path.join("results", args.workload.removesuffix(".yaml").removeprefix("workload-") if "-" in args.workload else "")
    os.makedirs(res_folder, exist_ok=True)
    cycle_path = os.path.join(res_folder, 'cycle-stat.yaml')
    arch_path = args.config
    mem_path = os.path.join(res_folder, 'mem-stat.yaml')
    comp_path = os.path.join(res_folder, 'comp-stat.yaml')
    work_path = args.workload

    cycle_stat = read_cycles(cycle_path, arch_path, work_path)
    workload_dic = extract_workload(work_path)
    mem_dic = comp_mem_energy(mem_path, cycle_stat, arch_path, res_folder)
    comp_dic = comp_computation_energy(comp_path, cycle_stat, arch_path, workload_dic, res_folder)
    
    print("SATA_Sim simulaton successes. Please go to the result folder to locate the energy results.")

