# simple script to delete incomplete runs from the 'saves' directory.
# WARNING: DO NOT run this script in the middle of running experiments, as those
# directories WILL GET DELETED!!!

import os, sys, shutil
import export_runs_csv as erc

if __name__ == "__main__":
    print("NOTE THAT RUNNING THIS SCRIPT WHEN EXPERIMENTS ARE RUNNING, CAN LEAD TO UNWANTED DELETION OF ACTIVE DIRECTORIES.")
    choice = input("Are you sure you want to clean up the saves directory right now? [y/N]:  ").strip().lower() or 'n'

    if choice not in ['y', 'n']:
        print(f"Unrecognized option {choice}... defaulting to `n`.")
        choice = 'n'

    if choice == 'n':
        print("Nothing was cleaned.")
        sys.exit(0)

    settings_files = erc.find_all('settings.yml', '.')
    run_paths = [os.path.dirname(f) for f in settings_files]

    delete_conds = [not (
        os.path.exists(os.path.join(run_path, 'results.json')) or
        os.path.exists(os.path.join(run_path, 'test_metrics.json')) or
        os.path.exists(os.path.join(run_path, 'metrics.pt')) or
        os.path.exists(os.path.join(run_path, 'models')) or
        os.path.exists(os.path.join(run_path, 'models', 'agg_client.pt')) or
        os.path.exists(os.path.join(run_path, 'agg_client.pt'))
    ) for run_path in run_paths]

    for path, delete in zip(run_paths, delete_conds):
        if delete and os.path.exists(path):
            print(f"Removing {path}...")
            shutil.rmtree(path) 
        elif not os.path.exists(path):
            print(f"Experiment dir {path} already removed")