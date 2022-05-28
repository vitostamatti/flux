# from .pipeline import Pipeline, Node
# from .data import DataCatalog
# from .data import PandasDataset

from typing import Union,List
from wrangler.base import AbstractDataset, AbstractTransformer
from wrangler.data.catalog import DataCatalog
from wrangler.data.datasets import PandasDataset

from wrangler.pipeline.pipeline import Pipeline
from wrangler.pipeline.node import Node

import pandas as pd

import dill

from wrangler.utils import load_dataset_object, write_config, read_config



class Wrangler():
    """
    A class to manage and orchestrate a DataCatalog and a Pipeline of Nodes.

    Attributes:
        data_catalog (DataCatalog, optional): The catalog from which Wrangler will load and save
            all data. Defaults to None. If None, Wrangler creates a default one with no data in it.
        pipeline (Pipeline, optional): The sequence of nodes which Wrangler will apply to the given
            datasets. Defaults to None. IF None, Wrangler creates a default one with no Nodes in it.

    Example:

        .. code-block:: python

            wrangler = Wrangler()
            cars = PandasDataset(data=data, name='cars')
            wrangler.add_dataset(cars)
            wrangler.add_node(
                name='first_node',
                transformer = ColumnDropper(column='useless_col'),
                inputs = 'cars',
                outputs = 'cars_output'
            )
            wrangler.fit_transform()

    """

    _default_dataset = 'intermediate'
    _default_pipeline = 'abt'

    def __init__(self, data_catalog:DataCatalog=None, pipeline:Pipeline=None) -> None:
        self.data_catalog = DataCatalog() if data_catalog is None else data_catalog
        self.pipeline = Pipeline() if pipeline is None else pipeline

        self._init()

    def _init(self):
        init_dataset = PandasDataset(self._default_dataset,pd.DataFrame())
        self.add_dataset(init_dataset)

    def add_dataset(self, dataset:AbstractDataset):
        """Register the given dataset with the given name to the data catalog
        of the wrangler. 

        Args:
            dataset (AbstractDataset): an instance of an implementation of AbstractDataset.
        """

        self.data_catalog.add(dataset)

    def add_node(self, transformer:AbstractTransformer,
                name:str=None, inputs:Union[str,List[str]]=None,
                outputs:Union[str,List[str]]=None):

        """Inserts a new node to the wrangler pipeline or overrides an
        existing node with the given name.

        Args:
            transformer (AbstractTransformer): an instance of an implementation of AbstractTransformer
                with its corresponding constructor parameters.
            name (str, optional): the name of the node and also identifier. Defaults to None.
            inputs (Union[str,List[str]], optional): the names of the datasets already registered
                to use as inputs. Defaults to None.
            outputs (Union[str,List[str]], optional): the names of the datasets to use as outputs. Defaults to None.
                If the outputs already exists, it overrides the data, otherwise, it creates a new dataset.
        """

        if not name:
            name = f"node_{len(self.pipeline.nodes)}"
        if not inputs:
            inputs = self._default_dataset
        if not outputs:
            outputs = self._default_dataset

        node = Node(name, transformer, inputs, outputs)

        self.pipeline.add(node)


    def fit_transform(self):
        """
        Calls the fit and the transform method of all internal nodes in a sequencial way.
        """
        self.pipeline.fit_transform(self.data_catalog)


    def transform(self):
        """
        Calls the transform method of all internal nodes in a sequencial way.
        """
        self.pipeline.transform(self.data_catalog)


    def fit(self):
        """Calls the fit and transform method of all previous nodes and the fit method
        of the last node in the pipeline.

        Note:
            Not implemented yet

        """
        pass

    def save(self, path:str):
        """
        It saves the data catalog references of all datasets that are not
        in memory (PandasDataset).

        It saves the sequence of nodes in the current pipeline.

        Args:
            path (str): destination path to save the object.
        """
        datasets = {}
        for name, dataset in self.data_catalog.datasets.items():
            if not isinstance(dataset, PandasDataset):
                datasets[name] = dataset

        objs = {
            "catalog": DataCatalog(datasets=datasets),
            "pipeline": self.pipeline,
        }
        with open(path + '.plk', 'wb') as f:
            dill.dump(objs, f)


    def load(self, path:str):
        """
        It loads the object from the given path and sets
        data catalog and pipeline attributes to the loaded ones.

        Args:
            path (str): origin path to load the object.
        """
        with open(path + '.plk' ,'rb') as f:
            objs = dill.load(f)

        self.pipeline = objs['pipeline']

        self.data_catalog = objs['catalog']

    def _get_datasets_params(self, list_of_datasets):
        datasets = {}
        for ds in list_of_datasets:
            if ds in self.data_catalog.datasets:
                ds_params = {}
                ds_params['type']=type(self.data_catalog.datasets[ds]).__name__
                if ds_params['type'] == 'PandasDataset':
                    ds_params['data'] = 'DataFrame'
                else:
                    for param, value in self.data_catalog.datasets[ds].__dict__.items():
                        if param == 'name':
                            continue
                        elif value is not None:
                            if isinstance(value, dict):
                                ds_params[param] = value
                            else:
                                ds_params[param] = str(value)
                datasets[ds] = ds_params
            else:
                ds_params = {}
                ds_params['type'] = 'PandasDataset'
                datasets[ds] = ds_params

        return datasets

    def datasets_to_config(self, path):
        data = {}
        datasets_inputs = self._get_datasets_params(self.pipeline.inputs())
        datasets_outputs = self._get_datasets_params(self.pipeline.outputs())

        data['inputs'] = datasets_inputs
        data['outputs'] = datasets_outputs

        write_config(path, data)
        return self


    def datasets_from_config(self, path):
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
            self.add_dataset(dataset)

        return self
