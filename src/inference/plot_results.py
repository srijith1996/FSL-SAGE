# -----------------------------------------------------------------------------
import os, sys
sys.path.append("../")
import json, yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils import plot_util as pu
from prettytable import PrettyTable

# -----------------------------------------------------------------------------
def setup(color_scheme=None):
    # maptlotlib setup
    pu.figure_setup()
    color_wheel = [
        "#4E79A7FF", "#F28E2BFF", "#E15759FF", "#76B7B2FF", "#59A14FFF",
        "#EDC948FF", "#B07AA1FF", "#FF9DA7FF", "#9C755FFF", "#BAB0ACFF"
    ]
    new_colors = []

    if color_scheme is not None:
        vals = np.unique(color_scheme)
        counts = {k: 0 for k in vals}
        for val in color_scheme:
            new_colors.append(
                pu.lighten_color(color_wheel[val], 0.85**counts[val])
            )
            counts[val] += 1

    else:
        new_colors = color_wheel

    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=new_colors)

# -----------------------------------------------------------------------------
def get_json_file(path):
    with open(path, 'r') as json_file:
        d = json.load(json_file)
    return d

# -----------------------------------------------------------------------------
def get_path(cfg, exp_cfg, key_name, path):
    if 'result_files' in exp_cfg.keys():
        file_name = f"{exp_cfg['result_files'][key_name]}.json"
    else:
        file_name = "results.json"

    name = os.path.join(path, file_name)
    if 'full_dirs' not in exp_cfg or not exp_cfg['full_dirs']:
        ind_to_brckt = key_name.find('[')
        if ind_to_brckt == -1: ind_to_brckt = key_name.find('(')
        if ind_to_brckt == -1: ind_to_brckt = len(key_name)

        key_name_only = key_name[:ind_to_brckt].strip()
        name = os.path.join(
            cfg["name_folder"][key_name_only],
            exp_cfg['model'],
            f"{exp_cfg['dataset']}-{exp_cfg['distribution']}",
            name
        )
    return os.path.join(cfg['prefix_dir'], name)

# -----------------------------------------------------------------------------
def accuracy_plot(
    save_dicts, metrics, metric_names, test_ids,
    x_comm_load=False, plots_dir='../plots', legend=True
):
    for metric, metric_name in zip(metrics, metric_names):
        fig_size = pu.get_fig_size(9, 7)
        fig = plt.figure(figsize=fig_size)

        ax = fig.add_subplot(111)
        key_names = []
        handles = []
        for i, (k, v) in enumerate(save_dicts.items()):
            if i in test_ids:
                if x_comm_load:
                    x_ = [c_load / (1024 ** 3) for c_load in v['comm_load'] ]
                else:
                    x_ = np.arange(0, len(v[metric]))
                label = k if legend != 'names_only' else None
                h, = ax.plot(x_, v[metric], label=label, lw=pu.plot_lw())
                key_name_only = k[:k.index('(')].strip() if '(' in k else k
                if key_name_only not in key_names:
                    key_names.append(key_name_only)
                    handles.append(h)

        ax.set_ylabel(metric_name)
        if x_comm_load:
            ax.set_xlabel("Communication Load (GB)")
        else:
            ax.set_xlabel("Rounds ($t$)")

        ax.set_axisbelow(True)

        if legend == 'names_only':
            ax.legend(handles, key_names)
        elif legend:
            ax.legend(loc='lower right')
        ax.grid(True, which='both', axis='both')
        plt.tight_layout()
        fpostfix = '_comm_load' if x_comm_load else ''
        fig.savefig(os.path.join(plots_dir, f'{metric}{fpostfix}.eps'))
        fig.savefig(os.path.join(plots_dir, f'{metric}{fpostfix}.png'))

# -----------------------------------------------------------------------------
def metrics_vs_dirichlet_alpha(
    save_dicts, metrics, metric_names, test_ids, plots_dir='../plots'
):
    '''
    save_dicts should be a nested dict in the form:
        {
            '<alg_name1>' : {
                <alpha_val1> : json_dict,
                <alpha_val2> : json_dict,
                ...
            },
            '<alg_name2>' : {
                <alpha_val1> : json_dict,
                <alpha_val2> : json_dict,
                ...
            }
        },
        
    '''
    os.makedirs(plots_dir, exist_ok=True)
    for metric, metric_name in zip(metrics, metric_names):
        fig_size = pu.get_fig_size(9, 7)
        fig = plt.figure(figsize=fig_size)

        ax = fig.add_subplot(111)
        x_axes = []
        metric_vals = []
        legends = []
        for i, (alg, alpha_dict) in enumerate(save_dicts.items()):
            if i in test_ids:
                x_axes.append(list(alpha_dict.keys()))
                metric_vals.append([v[metric][-1] for v in alpha_dict.values()])
                legends.append(alg)

        for x_, y_, leg_ in zip(x_axes, metric_vals, legends):
            ax.semilogx(x_, y_, label=leg_, lw=pu.plot_lw(), marker='o')
        
        ax.set_axisbelow(True)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(metric_name)
        ax.legend(loc='lower right')
        ax.grid(True, which='both', axis='both')
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f'{metric}_vs_dirichlet_alpha.png'))
        fig.savefig(os.path.join(plots_dir, f'{metric}_vs_dirichlet_alpha.eps'))

# -----------------------------------------------------------------------------
def misc_exps(config):
    for exp_name, exp in config['experiments'].items():
        print(exp_name)
        if 'type' in exp.keys() and exp['type'] == 'dirichlet_alpha':
            print(f"Plotting accuracy vs dirichlet alpha @ {exp_name}")

            def __get_path(alg_label, folder_name):
                final_path = os.path.join(
                    config['prefix_dir'],
                    config["name_folder"][alg_label],
                    exp['model'],
                    f"{exp['dataset']}-{exp['distribution']}",
                    folder_name, 'results.json'
                )
                return final_path

            def __strip_brackets(name):
                ind_to_brckt = name.find('[')
                if ind_to_brckt == -1: ind_to_brckt = name.find('(')
                if ind_to_brckt == -1: ind_to_brckt = len(name)
                return name[:ind_to_brckt].strip()

            setup()

            save_dict = {
                k: {
                    k_: get_json_file(__get_path(__strip_brackets(k), v_))
                    for k_, v_ in v.items()
                } for k, v in exp['save_locs'].items()
            }
            #print(save_dict)

            metrics_vs_dirichlet_alpha(
                save_dict, ["test_loss", "test_acc"],
                ["Test Loss", "Test Accuracy"], test_ids=exp['test_ids'],
                plots_dir=os.path.join(config['plots_dir'], exp_name)
            )

# -----------------------------------------------------------------------------
def main():
    with open('./exp_config.yaml', 'r') as yml_file:
        config = yaml.safe_load(yml_file)

    for exp_name, exp in config['experiments'].items():

        if 'disable' in exp.keys() and exp['disable']:
            print(f"Skipping plots for {exp_name}.")
            continue

        if 'type' in exp.keys():
            print(
                f"Experiments with `type` property will be plotted later. Skipping {exp_name} for now."
            )
            continue

        setup()
        if 'colorscheme' in exp.keys() and exp['colorscheme'] is not None:
            print(f"Creating new color scheme...")
            setup(color_scheme=exp['colorscheme'])

        print(f"Plotting for {exp["title"]}")
        results = {
            k : get_json_file(
                get_path(config, exp, k, v)
            ) for k, v in exp['save_locs'].items()
        }
        #print(results.keys())

        # create directory for plot
        plot_dir = os.path.join(config['plots_dir'], exp_name)
        os.makedirs(plot_dir, exist_ok=True)

        # plot 
        if config['plots']:
            accuracy_plot(
                results, ['test_acc', 'test_loss'],
                ['Test Accuracy', 'Test Loss'],
                test_ids=exp['test_ids'], plots_dir=plot_dir,
                legend=exp['legend'] if 'legend' in exp else True
            )
            accuracy_plot(
                results, ['test_acc', 'test_loss'],
                ['Test Accuracy', 'Test Loss'],
                test_ids=exp['test_ids'], x_comm_load=True,
                plots_dir=plot_dir,
                legend=exp['legend'] if 'legend' in exp else True
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

    misc_exps(config)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------