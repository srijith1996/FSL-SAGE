import numpy as np
import matplotlib.pyplot as plt
import plot_results as pr
from collections import deque

import os, sys, glob
sys.path.append("../")
from src.utils import plot_util as pu

ALIGN_INTERVAL_DIR_TIMESTAMP_10 = {
    2: '250330-225018',
    3: '250330-225018',
    5: '250330-225018',
    10: '250330-225018',
    20: '250330-225017',
    30: '250330-225018',
    40: '250330-225018',
    50: '250330-225017',
}
ALIGN_INTERVAL_DIR_TIMESTAMP_100 = {
    2: '250527-162637',
    3: '250525-210423',
    5: '250525-210423',
    10: '250525-210423',
    20: '250525-210422',
    30: '250525-210423',
    40: '250525-210423',
    50: '250525-210423',
}
PLOT_DIR = '../plots/icml/ablations'

def align_interval_ablation(save_dir, dataset='cifar10'):
    align_timestamp = ALIGN_INTERVAL_DIR_TIMESTAMP_10\
        if dataset == 'cifar10' else ALIGN_INTERVAL_DIR_TIMESTAMP_100
    final_test_accs = {
        k: np.max(pr.get_json_file(
            f'{save_dir}/R200m10E1B256q5l{k}-seed200/{v}/results.json'
        )['test_acc'])
        for k, v in align_timestamp.items()
    }
    fig_size = pu.get_fig_size(9, 7)
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    h, = ax.semilogx(
        final_test_accs.keys(), final_test_accs.values(),
        lw=pu.plot_lw()*0.8, markersize=4
    )

    if dataset == 'cifar10':
        ax.set_ylim([0.8, 0.9])
    ax.set_xlabel("Alignment interval ($l$)")
    ax.set_ylabel("Test Accuracy ($200$ rounds)")
    ax.grid(True, which='both', axis='both')
    plt.tight_layout()

    fig.savefig(f"{PLOT_DIR}/align_interval_{dataset}.eps")
    fig.savefig(f"{PLOT_DIR}/align_interval_{dataset}.png", dpi=200)

def aux_model_size_ablation(
    save_dir, dataset='cifar10', methods=['cse_fsl','fsl_sage']
):

    fig_size = pu.get_fig_size(9, 7)
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)

    aux_sizes_names = {
        0.005: 'resnet18_linear',
        3.522: 'resnet18_half',
        8.029: 'resnet18',
        40.078: 'resnet18_full'
    }

    for method in methods:
        method_accs = {
            k_: pr.get_json_file(
                f'{save_dir}/{method}/resnet18/{dataset}-iid/{v}/results.json'
            )['test_acc'][-1]
            for k_, v in aux_sizes_names.items()
        }
    
        ax.semilogx(
            method_accs.keys(), method_accs.values(),
            lw=pu.plot_lw()*0.8, markersize=4    
        )

    ax.set_xlabel("Auxiliary size (in MB)")
    ax.set_ylabel("Test Accuracy (@ $200$ GB)")
    ax.legend(['CSE-FSL', 'FSL-SAGE'])
    ax.grid(True, which='both', axis='both')
    plt.tight_layout()

    fig.savefig(f"{PLOT_DIR}/aux_size_{dataset}.eps")
    fig.savefig(f"{PLOT_DIR}/aux_size_{dataset}.png", dpi=200)

def get_times_from_log_file(file_path):

    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            if 'main.py:main()' in line and '>' in line:
                lines.append(line.rstrip())
            
    #print('\n'.join(lines))
    times_dict = {
        line.split(' ')[-3]: float(line.split(' ')[-1][:-1])
        for line in lines
    }

    times_dict['net'] = sum(list(times_dict.values()))
    return times_dict

def convert_hms(vals):
    hms_list = []
    for val in vals:
        h, m, s = int(val // 3600), int((val % 3600) // 60), int(val % 60)
        hms = ''
        if h == 0 and m == 0: hms = f'{s:d}s'
        elif h == 0:  hms = f'{m:d}m {s:d}s'
        else: hms = f'{h:d}h {m:d}m {s:d}s'
        hms_list.append(hms)

    return hms_list

def latency_analysis(
    bandwidth_mbps=20, save_dir='../saves', R=200, m=10, E=1, B=256, q=5, l=10,
    seed=200, methods=['fed_avg', 'sl_multi_server', 'sl_single_server',
    'cse_fsl', 'fsl_sage']
):
    dir_patterns = []
    for method in methods:
        if method == 'cse_fsl':
            dir_patterns.append(f'R{R}m{m}E{E}B{B}q{q}')
        elif method == 'fsl_sage':
            dir_patterns.append(f'R{R}m{m}E{E}B{B}q{q}l{l}')
        else:
            dir_patterns.append(f'R{R}m{m}E{E}B{B}')
            
    dir_names = [
        glob.glob(
            f'{save_dir}/{method}/resnet18/cifar10-iid/{dp}-seed{seed}/*'
        )[-1] for method, dp in zip(methods, dir_patterns)
    ]

    comm_lat_s = []
    comp_lat_s = []
    for method, dir_name in zip(methods, dir_names):
        net_comm_size_mb = pr.get_json_file(
            f'{dir_name}/results.json'
        )['comm_load'][-1] / (1000 ** 2)
        comm_lat_s.append(net_comm_size_mb / bandwidth_mbps)
        comp_lat_s.append(get_times_from_log_file(f'{dir_name}/output.log')['net'])

    print(comm_lat_s)
    print(comp_lat_s)

    fig, ax = plt.subplots(1, 1)
    plt.yscale('log')
    ax.bar(methods, comp_lat_s)
    ax.bar(methods, comm_lat_s, bottom=comp_lat_s)
    ax.set_ylim([10, None])
    ax.set_ylabel("Latency (s)", fontsize=1.5*pu.label_size())
    ax.grid(True, which='both', axis='y')

    ax.set_yticklabels(
        convert_hms(ax.get_yticks()),
        fontsize=1.5*pu.ticks_size(),
    )
    ax.set_xticklabels(
        ['FedAvg', 'SplitFed-MS', 'SplitFed-SS', 'CSE-FSL', 'FSL-SAGE'],
        fontsize=1.5*pu.ticks_size()
    )
    ax.legend(['Computation', 'Communication'])
    plt.tight_layout()

    fig.savefig(f"{PLOT_DIR}/latency.eps")
    fig.savefig(f"{PLOT_DIR}/latency.png", dpi=200)

if __name__ == "__main__":
    pr.setup()
    os.makedirs(PLOT_DIR, exist_ok=True)

    latency_analysis()
    #align_interval_ablation(save_dir='../results/align_ablations', dataset='cifar10')
    #align_interval_ablation(save_dir='../results/align_ablations_cifar100', dataset='cifar100')
    #aux_model_size_ablation(save_dir='../results/model_sizes_abalations/saves')
    #aux_model_size_ablation(save_dir='../results/model_sizes_abalations/saves', dataset='cifar100')