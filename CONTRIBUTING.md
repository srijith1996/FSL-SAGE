## Directory Structure
The directory structure looks like follows:
```
 - src
 |_ algos
    > contains FL algorithms
 |_ datasets
    > files to process/load datasets
 |_ models
    > pytorch models
   |_ aux_models
      > auxiliary models can be plugged in separately
 |_ utils
    > utility files
 |_ hydra_config
    > hierarchical configuration setup for hydra
```

## New algorithms, models and datasets
For contributing new models, algorithms and datasets add the files to the
respective folder.

- **Datasets:** A new dataset class for loading the dataset can be added to a
    new file in the [`datasets/`](src/datasets/) folder.  Finally add the dataset in the `get_dataset()` method in the respective [`__init__.py`](src/datasets/__init__.py).

- **Models:** For contributing a new model, place the client and server models
    anywhere in the [`models`](src/models/) folder, and call the
    `register_client_server_pair()` function with the `name`, `client` and
    `server` objects.

    - **Auxiliary models:**  To add an auxiliary model define the
        `AuxiliaryModel` or `GradScalarAuxiliaryModel` subclass for the
        auxiliary model in the [`aux_models`](src/models/aux_models/) folder,
        and add the `register_auxiliary_model` decorator. 

- **Algorithms:** Add the algorithm to a new or existing file in the
    [`algos`](src/algos/) folder and use the `register_algorithm` decorator to
    decorate the method.

Finally, you need to create a new `.yaml` file within the respective folder in
[`hydra_config`](src/hydra_config/).  The new model, dataset or algorithm can
then be run/tested using

```bash
python main.py model=... algorithm=... dataset=...
```