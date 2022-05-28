import yaml
import importlib
import sys

from wrangler import Wrangler
from wrangler.pipeline.node import Node
# sys.path.append(os.getcwd())
import os

import wrangler
def read_config(path):
    with open(os.path.join(path), 'r') as f:
        data = yaml.safe_load(f)
    return data

def write_config(path, data):
    with open(os.path.join(path), 'w') as f:
        yaml.dump(data,f)


def load_dataset_object(obj_name, obj_path="wrangler.data.datasets"):
    module_spec = importlib.util.find_spec(obj_path)
    module_obj = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module_obj)
    return getattr(module_obj, obj_name)   


def from_config(path, wrangler):
    config_data = read_config(path)

    data = {**config_data['inputs'], **config_data['outputs']}   
    list_of_datasets = []
    for dataset_name in data:
        dataset_conf = data[dataset_name]

        obj_name = dataset_conf.pop('type')
        obj_params = dataset_conf
        obj_params['name']=dataset_name

        dataset_class = load_dataset_object(obj_name)

        dataset = dataset_class(**obj_params)
        list_of_datasets.append(dataset)

    for dataset in list_of_datasets:
        wrangler.add_dataset(dataset)
    return wrangler
    
def to_config(path, wrangler):
    data = {}
    datasets_inputs = {}
    for ds in wrangler.pipeline.inputs():
        if ds in wrangler.data_catalog.datasets:
            ds_params = {}
            ds_params['type']=type(wrangler.data_catalog.datasets[ds]).__name__
            for param, value in wrangler.data_catalog.datasets[ds].__dict__.items():
                if value is not None:
                    ds_params[param] = value
            datasets_inputs[ds] = ds_params
        else:
            ds_params = {}
            ds_params['type'] = 'PandasDataset'
            datasets_inputs[ds] = ds_params

    data['inputs'] = datasets_inputs

    datasets_outputs = {}
    for ds in wrangler.pipeline.outputs():
        if ds in wrangler.data_catalog.datasets:
            ds_params = {}
            ds_params['type']=type(wrangler.data_catalog.datasets[ds]).__name__
            for param, value in wrangler.data_catalog.datasets[ds].__dict__.items():
                if value is not None:
                    ds_params[param] = value
            datasets_outputs[ds] = ds_params
        else:
            ds_params = {}
            ds_params['type'] = 'PandasDataset'
            datasets_outputs[ds] = ds_params

    data['outputs'] = datasets_outputs

    write_config(path, data)

# wrangler = Wrangler()
# wrangler.load('../../data/titanic_wrangler')
# path_to_config = '../../data/data_catalog_config_titanic.yml'
# to_config(path_to_config, wrangler)

# wrangler = Wrangler()
# path_to_config = '../../data/data_catalog_config_titanic.yml'
# from_config(path_to_config, wrangler)

# print(wrangler.data_catalog)

