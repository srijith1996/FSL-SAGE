## FSL-SAGE: Accelerating Federated Split Learning via Smashed Activation Gradient Estimation

### Introduction
Our Federated Split Learning (FSL) algorithm cuts down on communication
overheads in traditional Split Learning methods by directly estimating
server-returned gradients at each client using auxiliary models.  The auxiliary
models are much smaller versions of the server model which are explicitly
trained to estimate the gradients that the server model would return for the
client's local input.

The algorithm is summarized in the following schematic:
<div align="center">
<img src="./img/fsl_sage.png" alt="FSL-SAGE schematic"/>
</div>

### Requirements
The project requirements can be simply installed using the environment config
file [`conda_env.yaml`](conda_env.yaml) as follows:
```python
conda env create -f conda_env.yaml
```
which will create a conda environment by the name `sage`.  You can activate the
conda environment using:
```python
conda activate sage
```
and all dependency requirements should be met.

### Configuration
This project is powered by [Hydra](https://hydra.cc/docs/intro/), which allows
hierarchical configurations and easy running of multiple ML experiments.
The config files for hydra are located in the folder
[`hydra_config`](src/hydra_config).

There is a high degree of customizability here; datasets, models and FL
algorithms can be plugged in using configs.  Please check out our
[contributing readme](CONTRIBUTING.md) for more details.

### Datasets
Currently datasets are read and imported from the `datas` folder in the root of
the repository.  You can simply create a folder for the repository and download
the dataset there.  After performing the necessary preprocessing, simply
use/extend the `get_dataset()` function in [`datasets/__init__.py`](src/datasets/__init__.py)

### Training
To train FSL-SAGE with defaults from
[`config.yaml`](src/hydra_config/config.yaml), you can simply run
```bash
python main.py
```
Training results are saved in a `saves` folder in the root, where they will be
saved in folders segregated based on the FL algorithm, model, dataset and
distribution used.

The default number of clients (`num_clients`) is set to 10 and the default
number of rounds is `rounds=200`.
Each method will train upto the fixed `rounds` or until the number of MBs
specified in `comm_threshold_mb` is reached.

To choose a specific model or algorithm, the Hydra
[command-line override](https://hydra.cc/docs/advanced/override_grammar/basic/)
functionality can be used as follows
```bash
python main.py model=resnet18 algorithm=cse_fsl dataset=cifar100 dataset.distribution=iid
```
The following options are currently supported, click them to reveal the details:
<details>

<summary>Algorithm</summary>

**Syntax : `algorithm=<key>`**.
The FL algorithm to use for training.
List of algorithms currently supported:

|   Key              | Algorithm    |
|:-------------------|:-------------|
| `fed_avg`          | FedAvg       |
| `sl_multi_server`  | SplitFedv1   |
| `sl_single_server` | SplitFedv2   |
| `cse_fsl`          | CSE-FSL      |
| `fsl_sage`         | FSL-SAGE     |

</details>

<details>

<summary>Dataset</summary>

**Syntax : `dataset=<key>`**.
The dataset used in training.
List of datasets currently supported:

|   Key      | Dataset      |
|:-----------|:-------------|
| `cifar10`  | cifar10      |
| `cifar100` | cifar100     |

</details>

<details>

<summary>Model</summary>

**Syntax : `model=<key>`**.
The ML model to use for training.
List of models currently supported:

|   Key              | Model        |
|:-------------------|:-------------|
| `resnet18`         | ResNet-18    |
| `resnet50`         | ResNet-50    |
| `resnet56`         | ResNet-56    |
| `resnet110`        | ResNet-110   |

Note that currently the above resnet models apart from `resnet18` haven't been
tuned yet, so the results may not optimally represent FSL-SAGE's communication
benefits.

</details>

<details>

<summary>Data distribution</summary>

**Syntax : `dataset.distribution=<key>`**.
Determines the distrbution of the dataset across clients List of distributions
currently supported:

|   Key  | Distribution |
|:-------|:-------------|
| `iid`  | homogeneous  |
| `noniid_dirichlet`  | heterogeneous|

For `noniid_dirichlet` you can specify the value of `alpha` using the key
`dataset.alpha`, e.g., `dataset.alpha=1`.

</details>

\
We also support multiruns in parallel using the
[hydra-joblib-launcher](https://hydra.cc/docs/plugins/joblib_launcher/)
Thus, it is possible to run multiple experiments for different combinations of
hyperparams, models, datasets or algorithms given sufficient GPU memory.
```bash
python main.py -m model=resnet18,simple_conv algorithm=fed_avg,sl_single_server,sl_multi_server,cse_fsl,fsl_sage
```
The above would create parallel jobs that would run main.py on all combinations
of specified options.
The number of jobs can be controlled by modifying the `hydra.launcher.n_jobs`
option in [`config.yaml`](./src/hydra_config/config.yaml) or by specifying
`hydra.launcher.n_jobs=<jobs>` as an option to the script.

### Inference Plots
Please check out the [readme](./inference/README.md), the functions used in the
[`plot_results.py`](./inference/plot_results.py) and the configs in
[`exp_configs.yaml`](./inference/exp_config.yaml) on how to generate the plots
for accuracy, communication load, etc.