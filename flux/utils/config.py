import yaml
import importlib
import os

def read_config(path):
    if not path.endswith('.yml'):
        path = os.path.join(path, ".yml")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


def write_config(path, data):
    if not path.endswith('.yml'):
        path = os.path.join(path, ".yml")
    with open(path, "w") as f:
        yaml.dump(data, f)


def load_dataset_object(obj_name, obj_path="flux.data.datasets"):
    mod = importlib.import_module(obj_path)
    class_ = getattr(mod, obj_name)
    return class_