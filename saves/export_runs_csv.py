# This script finds the settings.yml file in all subdirectories, and maps the
# run paths to the different settings in a csv file

import os
import yaml
import pandas as pd

def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
    def ignore_unknown(self, node):
        return None 

SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)

if __name__ == "__main__":
    settings_files = sorted(find_all('settings.yml', '.'))
    run_paths = [os.path.dirname(f) for f in settings_files]
    
    all_settings = []
    col_names = set()
    for f in settings_files:
        with open(f, 'r') as fh:
            settings = yaml.load(fh, Loader=SafeLoaderIgnoreUnknown)
        setting = flatten_dict(settings)
        all_settings.append(setting)
        col_names.update(setting.keys())
    
    col_names = sorted(col_names)
    col_names.insert(0, 'run_paths')
    
    df = pd.DataFrame(columns=col_names)
    df['run_paths'] = run_paths
    
    for i, (path_name, setting) in enumerate(zip(run_paths, all_settings)):
        assert df.loc[i, 'run_paths'] == path_name
        for k, v in setting.items():
            if hasattr(v, '__iter__'): v = str(v)
            df.loc[i, k] = v
    
    df.to_csv('./run_summaries.csv')