# Class for Orchestrating de Datasets
from wrangler.data.datasets import PandasDataset, AbstractDataset
from typing import Any
from loguru import logger

class DataCatalog():
    '''

    Attributes:
        datasets (dict): names and implementations of an AbstractDataset.

    Example:

        .. code-block:: python

            cars_data = pd.DataFrame({
                "cars":["car1", "car2", "car3", "car4"],
            })
            cars = PandasDataset(data=cars_data, name='cars')
            catalog = DataCatalog()
            catalog.add(cars)
            df = catalog.load("cars")  

    '''

    def __init__(self, datasets=None) -> None:
        self.datasets = {} if datasets is None else datasets

    @property
    def _logger(self):
        return logger

    def load(self, name:str):
        '''Returns the data of the dataset name registered.
        Useful for inputs
        '''
        if name not in self.datasets:
            raise Exception(f"Dataset {name} not registered in data catalog")
        else:
            # self._logger.info(f"Loading dataset: {name}")
            dataset = self.datasets.get(name)
        return dataset.load()
        
    def save(self, name:str, data:Any):
        '''Saves de data to the dataset name registered.
        Useful for outputs
        '''
        # self._logger.info(f"Saving dataset: {name}")

        if name not in self.datasets:
            new_dataset = PandasDataset(name, data)
            self.add(new_dataset)
        else:
            dataset = self.datasets.get(name)
            dataset.save(data)

    def add(self, dataset:AbstractDataset):
        '''Register a new dataset with a given name.
        '''
        self._logger.info(f"Adding dataset: {dataset.name}")
        self.datasets[dataset.name] = dataset

    def release(self, name):
        '''Removes a dataset from the registered datasets.
        '''
        self._logger.info(f"Removing dataset: {name}")
        del self.datasets[name]

    def __str__(self) -> str:
        return str(self.datasets)

    def __repr__(self) -> str:
        return str(self.datasets)
        