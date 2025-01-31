# -----------------------------------------------------------------------------
import os, sys
sys.path.append("../")
from typing import List, Union
import json, yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from src.utils import plot_util as pu

# -----------------------------------------------------------------------------
def setup(color_scheme=None):
    # maptlotlib setup
    pu.figure_setup()
    #color_wheel = [
    #    "#4E79A7FF", "#F28E2BFF", "#E15759FF", "#76B7B2FF", "#59A14FFF",
    #    "#EDC948FF", "#B07AA1FF", "#FF9DA7FF", "#9C755FFF", "#BAB0ACFF"
    #]
    color_wheel = [
        "#446A99FF", "#D37A2FFF", "#C1494CFF", "#679E96FF", "#4F8A45FF",
        "#D6B73FFF", "#9A6991FF", "#E78C97FF", "#84674AFF", "#A09C95FF"
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

    markers = ['.', 'p', '*', 'd', 'x', 's', '+', 'P', '2', 'H']
    linestyles = [
        'dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), 'solid',
        (0, (1, 10)), (0, (5, 10)), (0, (5, 1)), (0, (3, 10, 1, 10, 1, 10)),
        (0, (3, 5, 1, 5))
    ]

    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=new_colors) + \
        mpl.cycler(marker=markers)

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
    save_dicts, metrics, metric_names, test_ids, metric_minimize,
    x_comm_load=False, plots_dir='../plots', legend=True,
    x_axis_rnds_lim=None, mark_at_metric=None, centralized_level=None
):

    for n, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        fig_size = pu.get_fig_size(9, 7)
        fig = plt.figure(figsize=fig_size)

        ax = fig.add_subplot(111)
        key_names = []
        handles = []
        plot_colors = []
        metric_vals = []
        curves = []
        x_limits = []
        for i, (k, v) in enumerate(save_dicts.items()):
            if i in test_ids:
                x_lim = len(v[0][metric])
                x_limits.append(x_lim)

                if x_axis_rnds_lim is not None:
                    x_lim = min(x_axis_rnds_lim, x_lim)

                v_ = v[0] if isinstance(v, list) else v
                x_ = [c_load / (1024 ** 3) for c_load in v_['comm_load']][:x_lim] \
                    if x_comm_load else np.arange(0, x_lim)

                #label = k if legend != 'names_only' else None
                plt_y = v[metric][:x_lim] if not isinstance(v, list) \
                    else np.mean(np.array([v_[metric] for v_ in v])[:, :x_lim], axis=0)
                curves.append(plt_y)

                h, = ax.plot(
                    x_, plt_y, lw=pu.plot_lw()*0.8,
                    markevery=(i/len(save_dicts.keys()) * 0.2, 0.2),
                    markersize=4
                )
                plot_colors.append(h.get_color())
                key_name_only = k[:k.index('(')].strip() if '(' in k else k

                if key_name_only not in key_names:
                    key_names.append(key_name_only)
                    handles.append(h)
                    metric_vals.append(np.max(plt_y))

        if metric == 'test_acc' and centralized_level is not None:
            #print(f"here {centralized_level}")
            xlims = ax.get_xlim()
            #print(xlims, [centralized_level]*2)
            ax.plot(
                xlims, [centralized_level]*2, color='blue', linestyle='dashed',
                linewidth=1.0, alpha=0.8, marker=''
            )
            ax.set_xlim(xlims)

        # plot horizontal and vertical lines indicating where each algorithm
        # attains chosen accuracy
        if metric == 'test_acc' and x_comm_load and mark_at_metric is not None:
            assert len(mark_at_metric) == len(metrics), \
                "`mark_at_metric` should have same length as metrics"

            assert len(metric_minimize) == len(metrics), \
                "`metric_minimize` should have same length as metrics"

            xlims = ax.get_xlim()
            ylims = ax.get_ylim()
            ax.plot(
                xlims, [mark_at_metric[n]]*2, linestyle='dashed', color='gray',
                linewidth=pu.plot_lw() * 0.8, marker=''
            )

            existing_pts = []
            for i, (k, v) in enumerate(save_dicts.items()):
                if not i in test_ids:
                    continue

                metric_arr = np.array(curves[i][:x_limits[i]])
                cond = (metric_arr <= mark_at_metric[n]) if metric_minimize[n] \
                    else (metric_arr >= mark_at_metric[n])
                mark_rnd_alg = np.argwhere(cond)[0][0] if np.any(cond) else None
                
                if mark_rnd_alg is not None:
                    v_ = v[0] if isinstance(v, list) else v
                    x_alg = v_['comm_load'][mark_rnd_alg] / (1024**3) \
                        if x_comm_load else mark_rnd_alg
                
                    # get 
                    ax.plot(
                        [x_alg]*2, [ylims[0], mark_at_metric[n]],
                        color=plot_colors[i], linestyle='dashed', linewidth=1.0,
                        marker=''
                    )

                    curr_pt = (x_alg + 1, mark_at_metric[n] - 0.3)
                    skip = False
                    for pt in existing_pts:
                        if np.abs(curr_pt[0] - pt[0]) <= 3.0:
                            skip = True

                    if not skip:
                        ax.text(
                            curr_pt[0], curr_pt[1],
                            f'{x_alg:.2f}GB' if x_comm_load else f'{x_alg:d}',
                            color=plot_colors[i], rotation=90, size=6, alpha=1.0,
                            weight='bold'
                        )
                        existing_pts.append(curr_pt)

            # restore original plot limits
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)

        ax.set_ylabel(metric_name)
        if x_comm_load:
            ax.set_xlabel("Communication Load (GB)")
        else:
            ax.set_xlabel("Rounds ($t$)")

        ax.set_axisbelow(True)

        #if legend == 'names_only':
        #    ax.legend(handles, key_names)
        if legend:
            idx = np.flip(np.argsort(metric_vals))
            handles = np.array(handles)[idx]
            key_names = np.array(key_names)[idx]

            if x_comm_load and metric == 'test_acc':
                arrow_locs = [
                    (20, 0.835), (80, 0.83), (110, 0.84), (130, 0.82), (150, 0.815)
                ]
                label_locs = [
                    (30, 0.71), (80, 0.71), (120, 0.67), (140, 0.63), (160, 0.57)
                ]
                #arrow_locs = [
                #    (20, 0.535), (80, 0.513), (110, 0.5), (135, 0.47), (160, 0.46)
                #]
                #label_locs = [
                #    (30, 0.41), (80, 0.41), (120, 0.37), (140, 0.33), (160, 0.29)
                #]
                arrow_coords = {name: loc for name, loc in zip(key_names, arrow_locs)}
                label_pos = {name: loc for name, loc in zip(key_names, label_locs)}
                for i, (label, position) in enumerate(label_pos.items()):
                    ax.annotate(
                        label, xy=arrow_coords[label], xytext=position,
                        bbox=dict(facecolor='white', edgecolor='white', boxstyle='round'),
                        arrowprops=dict(
                            edgecolor='black', facecolor='black',
                            arrowstyle="->", linewidth=1.0
                        ), fontsize=7, fontweight='bold', color=plot_colors[idx[i]],
                    )
            else:
                ax.legend(handles, key_names, fontsize=8)

        ax.grid(
            True, which='both', axis='both', linestyle='dotted', linewidth=0.5,
            color='gray', alpha=0.5
        )
        plt.tight_layout()
        fpostfix = '_comm_load' if x_comm_load else ''
        fig.savefig(os.path.join(plots_dir, f'{metric}{fpostfix}.eps'))
        fig.savefig(os.path.join(plots_dir, f'{metric}{fpostfix}.png'))

# -----------------------------------------------------------------------------
def metrics_vs_dirichlet_alpha(
    save_dicts: dict, metrics: List[str], metric_names: List[str],
    test_ids: List[int], metric_minimize: List[Union[bool, None]],
    align_at: str='round', align_at_val: float=None,
    plots_dir='../plots/default'
):
    '''Plot specified metrics vs. Dirichlet alpha.

    Plots the specified metrics in `metric_names` with respect to the parameter
    alpha of a Dirichlet distribution.  For each metric, the minimum, maximum or
    the last value upto the round determined by `align_at` and `align_at_val`,
    are plotted.

    Params
    ------
    save_dicts - dict
        should be a nested dict in the form:
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
        }
        where the `json_dict` corresponds to the run metrics as a dictionary
        with keys as metric names and values as a list of metric value collected
        at each communication round.  One of the keys in the `json_dict`s must be
        'comm_load'.

    metrics - List[str]
        list of metric key names which match the keys in the above `json_dict`s.

    metric_names - List[str]
        list of full metric names corresponding to the entries in metrics above.
        This is used for the y-label of the plots.

    test_ids - List[int]
        The `alg_names` in the save_dict indexed by `test_ids` will be plotted.

    metric_minimize - List of {True, False, None} values
        Indicates whether the 'best' metric is obtained by minimizing or
        maximizing the list of metric values in the `json_dict`.  If the entry
        in `metric_minimize` corresponding to `metrics` is None, the last entry
        of the aligned metric list is picked.

    align_at - optional, one of {'round', 'comm_load'}
        Indicates whether to use the number of rounds or the communication load
        spent, as the basis of comparing different algorithms.

    align_at_val - optional, int or float
        Behaviour depends on the value of `align_at`...
          - align_at == 'round':
            -- align_at_val is None: pick min, max and final values upto the
               last round of each algorithm.
            -- align_at_val == x   : pick the min, max and final values upto the
               `x^{th}` round of each algorithm.
          - align_at == 'comm_load':
            -- align_at_val is None: pick min, max and final values upto the
               minimum last communication load value across all algorithms.  For
               e.g., if fedavg consumes `2a GB` and fsl_sage consumes `a GB`,
               then the align point is taken at `a GB`.
            -- align_at_val == a   : pick the min, max and final values upto the
               `a GB` point of the communication load for each algorithm.

    plots_dir - Save `.eps` and `.png` plots for this experiment at this
        location.
        
    '''
    
    os.makedirs(plots_dir, exist_ok=True)
    if metric_minimize is None: metric_minimize = [None] * len(metrics)

    align_idx = []
    if align_at == 'comm_load':
        align_comm_load = min([
            min([v['comm_load'][-1] for v in alg_runs.values()]) \
                for alg_runs in save_dicts.values()
        ]) if align_at_val is None else align_at_val * (1024**3)
        align_idx = [[
            (np.array(v['comm_load']).searchsorted(
                align_comm_load, side='right') - 1) \
                for v in alg_runs.values()
            ] for alg_runs in save_dicts.values()
        ]
    elif align_at == 'round':
        if align_at_val is None:
            align_idx = [[
                    len(v['comm_load'])-1 for v in alg_runs.values()
                ] for alg_runs in save_dicts.values()
            ]
        else:
            align_idx = [[align_at_val for _ in alg_runs.values()] \
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
        max_met_val = []
        for i, (alg, alpha_dict) in enumerate(save_dicts.items()):
            if i in test_ids:
                x_axes.append(list(alpha_dict.keys()))
                curr_met_list = [
                    __get_val(i, j, v, metric, metric_min)
                     for j, v in enumerate(alpha_dict.values())
                ]
                metric_vals.append(curr_met_list)
                legends.append(alg)
                max_met_val.append(np.max(curr_met_list))

        handles = []
        for x_, y_ in zip(x_axes, metric_vals):
            h, = ax.semilogx(x_, y_, lw=pu.plot_lw(), marker='o')
            handles.append(h)
        
        ax.set_axisbelow(True)
        ax.set_xlabel(r"$\alpha$")
        ylabel = f'{metric_name} @ ${align_comm_load/(1024**3):.2f}$ GiB' \
            if align_at == 'comm_load' else f'{metric_name}'
        ax.set_ylabel(ylabel)

        idx = np.flip(np.argsort(max_met_val))
        handles, legends = np.array(handles)[idx], np.array(legends)[idx]
        ax.legend(handles, legends, loc='lower right')

        ax.grid(True, which='both', axis='both')
        plt.tight_layout()
        met_type = 'best' if metric_min is not None else 'final'
        fig.savefig(os.path.join(
            plots_dir, f'{metric}_{met_type}_{align_at}' +
            f'_{align_at_val}_vs_dirichlet_alpha.png'
        ))
        fig.savefig(os.path.join(
            plots_dir, f"{metric}_{met_type}_{align_at}" +
            f"_{align_at_val if align_at_val else 'default'}" +
            f"_vs_dirichlet_alpha.eps"
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
        ax.set_xlim([ax.get_xlim()[0], ax.get_xlim()[1] + 0])
        # produce a legend with the unique colors from the scatter
        legend1 = ax.legend(
            handles, labels, loc="lower left", handletextpad=0.0
        )
        ax.add_artist(legend1)

        # produce a legend with a cross-section of sizes from the scatter
        handles, labels = h_.legend_elements(
            prop="sizes", alpha=0.6, num=[1e-2, 1e0, 1e4], fmt='{x:.1e}',
            func=s_func_inv
        )
        legend2 = ax.legend(
            handles, labels, loc="lower right", handletextpad=0.0,
            title=r'$\alpha$', frameon=False
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

        print(exp['type'])
        for exp_type in exp['type'].split(','):
            if exp_type == 'dirichlet_alpha':
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
                    align_at_val=80.0,
                    plots_dir=os.path.join(config['plots_dir'], exp_name)
                )

                metrics_vs_dirichlet_alpha(
                    save_dict, ["test_loss", "test_acc"],
                    ["Test Loss", "Test Accuracy"], test_ids=exp['test_ids'],
                    metric_minimize=[True, False], align_at='comm_load',
                    align_at_val=50.0,
                    plots_dir=os.path.join(config['plots_dir'], exp_name)
                )

                metrics_vs_dirichlet_alpha(
                    save_dict, ["test_loss", "test_acc"],
                    ["Test Loss", "Test Accuracy"], test_ids=exp['test_ids'],
                    metric_minimize=[True, False], align_at='round',
                    align_at_val=200,
                    plots_dir=os.path.join(config['plots_dir'], exp_name)
                )

                #metrics_vs_dirichlet_alpha(
                #    save_dict, ["test_loss", "test_acc"],
                #    ["Test Loss", "Test Accuracy"], test_ids=exp['test_ids'],
                #    metric_minimize=[None, None], align_at='round',
                #    plots_dir=os.path.join(config['plots_dir'], exp_name)
                #)

            elif exp_type == 'metric_comm_scatter':
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

            else:
                raise Exception(f"Unknown experiment type {exp_type}")

# -----------------------------------------------------------------------------
def make_table(exp_name, exp, results, choose_fn=lambda x: -1, suffix='final'):
    table = PrettyTable()
    table.field_names = ['', 'Acc', 'load']

    def get_entries(v):
        if isinstance(v, list):
            acc_mat = np.array([v_['test_acc'] for v_ in v])
            acc_mean = np.mean(acc_mat, axis=0)
            acc_std = np.std(acc_mat, axis=0)
            idx = choose_fn(acc_mean)
            comm_load = v[0]['comm_load'][idx] / (1024**3)
            acc = acc_mean[idx] * 100.0
        else:
            idx = choose_fn(acc_mean)
            comm_load = v['comm_load'][idx] / (1024**3)
            acc = v['test_acc'][idx] * 100.0
        return acc, comm_load

    for i, (k, v) in enumerate(results.items()):
        if i in exp['test_ids']:
            acc, load = get_entries(v)
            table.add_row([k, f'{acc:.2f}', f'{load:.2f}'])

    print(table)
    os.makedirs(exp_name, exist_ok=True)
    with open(os.path.join(exp_name, f'table_{suffix}.txt'), 'w') as f:
        print(table, file=f, flush=True)

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
        results = {}
        for k, v in exp['save_locs'].items():
            if isinstance(v, list):
                results[k] = [get_json_file(
                    get_path(config, exp, k, v_)) for v_ in v
                ]
            else:
                results[k] = get_json_file(get_path(config, exp, k, v))
        #results = {
        #    k : get_json_file(
        #        get_path(config, exp, k, v)
        #    ) for k, v in exp['save_locs'].items()
        #}
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
                test_ids=exp['test_ids'],
                metric_minimize=[False, True], 
                plots_dir=plot_dir,
                legend=exp['legend'] if 'legend' in exp else True,
                x_axis_rnds_lim=x_axis_rnds_lim,
                mark_at_metric=[exp['accuracy_mark_level'], exp['loss_mark_level']],
                centralized_level=exp['centralized_level'] if 'centralized_level' in exp else None
            )
            accuracy_plot(
                results, ['test_acc', 'test_loss'],
                ['Test Accuracy', 'Test Loss'],
                test_ids=exp['test_ids'], x_comm_load=True,
                metric_minimize=[False, True], 
                plots_dir=plot_dir,
                legend=exp['legend'] if 'legend' in exp else True,
                x_axis_rnds_lim=x_axis_rnds_lim,
                mark_at_metric=[exp['accuracy_mark_level'], exp['loss_mark_level']],
                centralized_level=exp['centralized_level'] if 'centralized_level' in exp else None
            )

        # table
        if config['table']:
            make_table(exp_name, exp, results, choose_fn=np.argmax, suffix='best')
            make_table(exp_name, exp, results)

    misc_exps(config)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------