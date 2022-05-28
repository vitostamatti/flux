import sys
import itertools
import inspect
import yaml
import importlib
import os


def _get_imports(module):
    import inspect
    predicate = lambda x: inspect.ismodule or inspect.isfunction or inspect.isbuiltin or inspect.isclass
    imports = []
    for x,y in inspect.getmembers(module, predicate):
        if inspect.ismodule(y):
            if x.find("__")==-1 and y.__name__.find("__") == -1:
                imports.append((x, y.__name__))
        elif inspect.isfunction(y) or inspect.isclass(y):
            if y.__module__ != "__main__" and y.__module__ != sys.modules[__name__].__name__:
                imports.append((x, y.__name__, y.__module__))
        elif inspect.isbuiltin(y):
            if y.__self__ != "__main__":
                if y.__self__.__name__ != "builtins":
                    imports.append((x, y.__name__, y.__self__.__name__))
                # imports.append((x, y.__name__, y.__self__))

    import_execs = []
    for imp in imports:
        if len(imp)==2:
            if imp[0]==imp[1]:
                import_execs.append(f"import {imp[1]}")
            else:
                import_execs.append(f"import {imp[1]} as {imp[0]}")
        if len(imp)==3:
            if imp[0]==imp[1]:
                import_execs.append(f"from {imp[2]} import {imp[0]}")
            else:
                import_execs.append(f"from {imp[2]} import {imp[0]} as {imp[1]}")

    return import_execs



# class wrangler_func(object):
#     def __init__(self, f) -> None:
#         self.f = f
#         self.source = self._get_source(f)
    
#     def _get_source(self, f):
#         import_execs = _get_imports(sys.modules[f.__module__])
#         source = list(itertools.dropwhile(
#             lambda line: line.startswith('@'), inspect.getsource(f).splitlines()
#             ))
        
#         for idx, imp in enumerate(import_execs):
#             source.insert(1+idx, f"    {imp}")
        
#         return source

#     def __call__(self, *args, **kwargs):
#         exec('\n'.join(self.source))
#         fn_name = self.f.__name__ 
#         return eval(f"{fn_name}(*args, **kwargs)")


# def wrangler_func(f):
#     def wrapper(*args, **kwargs):
#         import_execs = _get_imports(sys.modules[f.__module__])
#         source = list(itertools.dropwhile(
#             lambda line: line.startswith('@'), inspect.getsource(f).splitlines()
#             ))
        
#         for idx, imp in enumerate(import_execs):
#             source.insert(1+idx, f"    {imp}")

#         exec('\n'.join(source))
#         fn_name = f.__name__ 
#         print('\n'.join(source))
#         return eval(f"{fn_name}(*args, **kwargs)")
#     return wrapper



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