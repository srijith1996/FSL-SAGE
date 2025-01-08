# -----------------------------------------------------------------------------
import os, sys
sys.path.append("../")
from typing import List, Union
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
    x_comm_load=False, plots_dir='../plots', legend=True,
    x_axis_rnds_lim=None
):
    for metric, metric_name in zip(metrics, metric_names):
        fig_size = pu.get_fig_size(9, 7)
        fig = plt.figure(figsize=fig_size)

        ax = fig.add_subplot(111)
        key_names = []
        handles = []

        for i, (k, v) in enumerate(save_dicts.items()):
            if i in test_ids:
                x_lim = len(v[metric])

                if x_axis_rnds_lim is not None:
                    x_lim = min(x_axis_rnds_lim, x_lim)

                x_ = [c_load / (1024 ** 3) for c_load in v['comm_load']][:x_lim] \
                    if x_comm_load else np.arange(0, x_lim)

                label = k if legend != 'names_only' else None
                h, = ax.plot(x_, v[metric][:x_lim], label=label, lw=pu.plot_lw())
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
    save_dicts: dict, metrics: List[str], metric_names: List[str],
    test_ids: List[int], metric_minimize: List[Union[bool, None]],
    align_at: str='round', plots_dir='../plots/default'
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

    if metric_minimize is None: metric_minimize = [None] * len(metrics)

    align_idx = []
    if align_at == 'comm_load':
        min_comm_load = min([
            min([v['comm_load'][-1] for v in alg_runs.values()]) \
                for alg_runs in save_dicts.values()
        ])
        align_idx = [[
            (np.array(v['comm_load']).searchsorted(min_comm_load, side='right') - 1) \
                for v in alg_runs.values()
            ] for alg_runs in save_dicts.values()
        ]
    elif align_at == 'round':
        align_idx = [[len(v['comm_load'])-1 for v in alg_runs.values()] \
            for alg_runs in save_dicts.values()
        ]
    else:
        raise Exception(
            f"Unknown value `{align_at}` passed to `align_at`; expected `round` or `comm_load`."
        )

    #print("Indices for alignment:-")
    #for alg, alg_val, align_arr in zip(save_dicts.keys(), save_dicts.values(), align_idx):
    #    for alp, al_id in zip(alg_val.keys(), align_arr):
    #        print(f"  - {alg}(alpha={alp:.2g}) : {al_id:4d}")

    def __get_val(i, j, val_list, metric, metric_min):
        reduced_met_list = val_list[metric][:(align_idx[i][j] + 1)]
        val = reduced_met_list[-1] if metric_min is None else (
            np.min(reduced_met_list) if metric_min else \
                np.max(reduced_met_list)
        )
        return val
        
    for metric, metric_name, metric_min in \
        zip(metrics, metric_names, metric_minimize):

        fig_size = pu.get_fig_size(9, 7)
        fig = plt.figure(figsize=fig_size)

        ax = fig.add_subplot(111)
        x_axes = []
        metric_vals = []
        legends = []
        for i, (alg, alpha_dict) in enumerate(save_dicts.items()):
            if i in test_ids:
                x_axes.append(list(alpha_dict.keys()))
                metric_vals.append([
                    __get_val(i, j, v, metric, metric_min)
                     for j, v in enumerate(alpha_dict.values())
                ])
                legends.append(alg)

        for x_, y_, leg_ in zip(x_axes, metric_vals, legends):
            ax.semilogx(x_, y_, label=leg_, lw=pu.plot_lw(), marker='o')
        
        ax.set_axisbelow(True)
        ax.set_xlabel(r"$\alpha$")
        ylabel = f'{metric_name} @ ${min_comm_load/(1024**3):.2f}$ GiB' \
            if align_at == 'comm_load' else metric_name
        ax.set_ylabel(ylabel)
        ax.legend(loc='lower right')
        ax.grid(True, which='both', axis='both')
        plt.tight_layout()
        met_type = 'best' if metric_min is not None else 'final'
        fig.savefig(os.path.join(
            plots_dir, f'{metric}_{met_type}_{align_at}_vs_dirichlet_alpha.png'
        ))
        fig.savefig(os.path.join(
            plots_dir, f'{metric}_{met_type}_{align_at}_vs_dirichlet_alpha.eps'
        ))

# -----------------------------------------------------------------------------
def metrics_vs_comm_load_scatter(
    save_dicts: dict, metrics: List[str], metric_names: List[str],
    test_ids: List[int], metric_minimize: List[Union[bool, None]] = None, 
    plots_dir: str='../plots/default'
):
    '''Scatter plot of accuracy vs communication load.
    
    Scatter plot of best or final accuracy vs the corresponding communication
    load for various algorithms at different values for non-iid sampling using
    the Dirichlet distribution.

    Params
    ------
    save_dicts: dict
        A nested dictionary mapping to the run results parsed from a JSON file.
        The JSON file dictionary contains mappings from metrics -> arrays
        containing the values of the metrics at various rounds.

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

    metrics: List[str]
        the namestrings of different metrics to be plotted.  For e.g.,
        'test_loss', 'test_acc', etc.  These should correspond to the key names
        in the JSON file.

    metric_names: List[str]
        This will appear as the y-axis label in the final plot for the
        corresponding metric.  Should be of the same length as `metrics`.
    
    test_ids: List[int]
        The algorithms within save_dicts that should be plotted.

    metric_minimize: optional, List[Union[bool, None]] = None
        A boolean array of the same length as `metrics`, indicates whether the
        corresponding metric is considered "better" when minimized.  Can be
        either `True`, `False` or `None`. If None, the last value of the metric
        is chosen.  By default all values are None, and the last metric value is
        considered.

    plots_dir: str
        Location of the directory to save the current plots in.

    '''

    os.makedirs(plots_dir, exist_ok=True)

    if metric_minimize is None: metric_minimize = [None] * len(metrics)

    s_func = lambda x: np.pi * (((-np.log10(x) + 5) / 9) * 3 + 1)**2
    s_func_inv = lambda x: 10**(-(((np.sqrt(x / np.pi) - 1) / 3 * 9) - 5))
        
    for metric, metric_name, metric_min in \
        zip(metrics, metric_names, metric_minimize):

        plt_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        fig_size = pu.get_fig_size(9, 7)
        fig = plt.figure(figsize=fig_size)

        ax = fig.add_subplot(111)
        x, y, c, s = [], [], [], []
        handles, labels = [], []
        ct = 0
        for i, (alg, alpha_dict) in enumerate(save_dicts.items()):
            if not i in test_ids: continue

            idx = np.array([
                -1 if metric_min is None else (
                    np.argmin(v[metric]) if metric_min \
                        else np.argmax(v[metric])
                ) for v in alpha_dict.values()
            ])
            x_ = [
                v['comm_load'][i_] / (1024**3) for i_, v in zip(idx,
                alpha_dict.values())
            ]
            y_ = [
                v[metric][i_] for i_, v in zip(idx, alpha_dict.values())
            ]
            r = np.array(list(alpha_dict.keys()))
            s_ = [s_func(r_) for r_ in r]
            x.extend(x_)
            y.extend(y_)
            s.extend(s_)

            h_ = ax.scatter(x_, y_, s=s_, alpha=0.9)
            handles.append(h_)
            labels.append(alg)

            ct += 1

        ax.set_axisbelow(True)
        ax.set_xlabel("Communication Load (GB)")
        ax.set_ylabel(metric_name)
        ax.set_xlim([ax.get_xlim()[0], ax.get_xlim()[1] + 50])
        # produce a legend with the unique colors from the scatter
        legend1 = ax.legend(
            handles, labels, loc="lower right", handletextpad=0.0
        )
        ax.add_artist(legend1)

        # produce a legend with a cross-section of sizes from the scatter
        handles, labels = h_.legend_elements(
            prop="sizes", alpha=0.6, num=[1e-2, 1e0, 1e4], fmt='{x:.1e}',
            func=s_func_inv
        )
        legend2 = ax.legend(
            handles, labels, loc="upper right", handletextpad=0.0, title=r'$\alpha$'
        )
        
        #ax.legend(loc='lower center')
        ax.grid(True, which='both', axis='both')
        plt.tight_layout()
        met_type = 'best' if metric_min is not None else 'final'
        fig.savefig(os.path.join(
            plots_dir, f'{metric}_{met_type}_vs_commload_scatter.png'
        ))
        fig.savefig(os.path.join(
            plots_dir, f'{metric}_{met_type}_vs_commload_scatter.eps'
        ))

# -----------------------------------------------------------------------------
def misc_exps(config):
    for exp_name, exp in config['experiments'].items():

        if 'disable' in exp and exp['disable']:
            print(f"\033[93mSkipping plots for {exp_name}.\033[0m")
            continue

        if not 'type' in exp.keys(): continue

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

        if exp['type'] == 'dirichlet_alpha':
            print(f"\033[92mPlotting accuracy vs dirichlet alpha @ {exp_name}\033[0m")

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
                metric_minimize=[True, False], align_at='comm_load',
                plots_dir=os.path.join(config['plots_dir'], exp_name)
            )

            metrics_vs_dirichlet_alpha(
                save_dict, ["test_loss", "test_acc"],
                ["Test Loss", "Test Accuracy"], test_ids=exp['test_ids'],
                metric_minimize=[None, None], align_at='comm_load',
                plots_dir=os.path.join(config['plots_dir'], exp_name)
            )

        if exp['type'] == 'metric_comm_scatter':
            print(f"\033[92mPlotting best metric vals vs best communication load @ {exp_name}\033[0m")

            save_dict = {
                k: {
                    k_: get_json_file(__get_path(__strip_brackets(k), v_))
                    for k_, v_ in v.items()
                } for k, v in exp['save_locs'].items()
            }
            #print(save_dict)

            metrics_vs_comm_load_scatter(
                save_dict, ["test_loss", "test_acc"],
                ["Test Loss", "Test Accuracy"], test_ids=exp['test_ids'],
                metric_minimize=[True, False],
                plots_dir=os.path.join(config['plots_dir'], exp_name)
            )

            metrics_vs_comm_load_scatter(
                save_dict, ["test_loss", "test_acc"],
                ["Test Loss", "Test Accuracy"], test_ids=exp['test_ids'],
                metric_minimize=[None, None],
                plots_dir=os.path.join(config['plots_dir'], exp_name)
            )

# -----------------------------------------------------------------------------
def main():
    with open('./exp_config.yaml', 'r') as yml_file:
        config = yaml.safe_load(yml_file)

    for exp_name, exp in config['experiments'].items():

        if 'disable' in exp.keys() and exp['disable']:
            print(f"\033[93mSkipping plots for {exp_name}.\033[0m")
            continue

        if 'type' in exp.keys():
            print(
                f"\033[94mExperiments with `type` property will be plotted later. Skipping {exp_name} for now.\033[0m"
            )
            continue

        setup()
        if 'colorscheme' in exp.keys() and exp['colorscheme'] is not None:
            print(f"Creating new color scheme...")
            setup(color_scheme=exp['colorscheme'])

        print(f"\033[92mPlotting for {exp["title"]}\033[0m")
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
            x_axis_rnds_lim = exp['rounds_till'] \
                if 'rounds_till' in exp else None
            accuracy_plot(
                results, ['test_acc', 'test_loss'],
                ['Test Accuracy', 'Test Loss'],
                test_ids=exp['test_ids'], plots_dir=plot_dir,
                legend=exp['legend'] if 'legend' in exp else True,
                x_axis_rnds_lim=x_axis_rnds_lim
            )
            accuracy_plot(
                results, ['test_acc', 'test_loss'],
                ['Test Accuracy', 'Test Loss'],
                test_ids=exp['test_ids'], x_comm_load=True,
                plots_dir=plot_dir,
                legend=exp['legend'] if 'legend' in exp else True,
                x_axis_rnds_lim=x_axis_rnds_lim
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