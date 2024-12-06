import os
import json, yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils import plot_util as pu
from prettytable import PrettyTable

def setup():
    # maptlotlib setup
    pu.figure_setup()
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[
        "#4E79A7FF", "#F28E2BFF", "#E15759FF", "#76B7B2FF", "#59A14FFF",
        "#EDC948FF", "#B07AA1FF", "#FF9DA7FF", "#9C755FFF", "#BAB0ACFF"
    ])

def get_json_file(path):
    with open(path, 'r') as json_file:
        d = json.load(json_file)
    return d

def accuracy_plot(
    save_dicts, metrics, metric_names, test_ids,
    x_comm_load=False, plots_dir='../plots',
):
    for metric, metric_name in zip(metrics, metric_names):
        fig_size = pu.get_fig_size(9, 7)
        fig = plt.figure(figsize=fig_size)

        ax = fig.add_subplot(111)
        for i, (k, v) in enumerate(save_dicts.items()):
            if i in test_ids:
                if x_comm_load:
                    x_ = [c_load / (1024 ** 3) for c_load in v['comm_load'] ]
                else:
                    x_ = np.arange(0, len(v[metric]))
                ax.plot(x_, v[metric], label=k, lw=pu.plot_lw())

        ax.set_ylabel(metric_name)
        if x_comm_load:
            ax.set_xlabel("Communication Load (GB)")
        else:
            ax.set_xlabel("Rounds ($t$)")

        ax.set_axisbelow(True)

        ax.legend(loc='lower right')
        ax.grid(True, which='both', axis='both')
        plt.tight_layout()
        fpostfix = '_comm_load' if x_comm_load else ''
        fig.savefig(os.path.join(plots_dir, f'{metric}{fpostfix}.eps'))
        fig.savefig(os.path.join(plots_dir, f'{metric}{fpostfix}.png'))
        
def main():
    with open('./exp_config.yml', 'r') as yml_file:
        config = yaml.safe_load(yml_file)

    for exp_name, exp in config['experiments'].items():
        results = {
            k : get_json_file(os.path.join(
                config['prefix_dir'], v, f"{exp['result_files'][k]}.json"
            )) for k, v in exp['save_locs'].items()
        }
        #print(results.keys())

        # create directory for plot
        plot_dir = os.path.join(config['plots_dir'], exp_name)
        os.makedirs(plot_dir, exist_ok=True)

        # plot 
        if config['plots']:
            accuracy_plot(
                results, ['test_acc', 'test_loss'], ['Test Accuracy', 'Test Loss'],
                test_ids=exp['test_ids'], plots_dir=plot_dir
            )
            accuracy_plot(
                results, ['test_acc', 'test_loss'], ['Test Accuracy', 'Test Loss'],
                test_ids=exp['test_ids'], x_comm_load=True, plots_dir=plot_dir
            )

        # table
        if config['table']:
            table = PrettyTable()
            table.field_names = ['', 'Acc', 'load']
            for i, (k, v) in enumerate(results.items()):
                if i in exp['test_ids']:
                    load = v['comm_load'][-1] / (1024**3)
                    acc = v['test_acc'][-1] * 100.0
                    table.add_row([k, f'{acc:.2f}', f'{load:.2f}'])
            print(table)

if __name__ == "__main__":
    setup()
    main()