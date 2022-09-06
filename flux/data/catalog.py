# Class for Orchestrating de Datasets
from flux.data.datasets import MemoryDataset, AbstractDataset

from typing import Any,Dict, Union, List
import logging
import pprint



class DataCatalog(object):
    '''
    Object used to register, load and save a collection of datasets

    Attributes:
        datasets (dict): names and implementations of an AbstractDataset.

    Example:

        .. code-block:: python

            catalog = DataCatalog()
            dataset = MemoryDataset()
            catalog.add(
                name = "mydataset",
                dataset = dataset
            )
    '''

    def __init__(self, datasets:Dict[str,AbstractDataset]=None) -> None:
        self._datasets = {} if datasets is None else datasets

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    @property
    def datasets(self):
        return self._datasets
        

    def load(self, names:Union[str,List[str]])->Dict:
        """
        Returns the data of the dataset name registered.
        Useful for inputs

        Raises:
            Exception: if the any of the names provided is not registered in the catalog

        Returns:
            Dict: a dictionary of {name:dataset}.
        """
        if isinstance(names,str):
            if names not in self.datasets:
                msg = f"Dataset {names} not registered in data catalog"
                self._logger.error(msg)
                raise Exception(msg)
            else:
                # self._logger.info(f"Loading dataset: {name}")
                data = {names: self.datasets[names].load()}
            return data
        elif isinstance(names, list):
            data = {}
            for n in names:
                if n not in self.datasets:
                    msg = f"Dataset {n} not registered in data catalog"
                    self._logger.error(msg)
                    raise Exception(msg)
                data[n]=self.datasets[n].load()
            return data

        
    def save(self, name:str, data:Any):
        """
        Saves de data to the dataset name registered.
        Useful for outputs

        Args:
            name (str): name of the dataset to save.
            data (Any): data to save in the dataset.
        """
        self._logger.info(f"Saving dataset: {name}")

        if name not in self.datasets:
            new_dataset = MemoryDataset(data)
            self.add(name, new_dataset)
        else:
            dataset = self.datasets.get(name)
            dataset.save(data)


    def add(self,  name:str, dataset:AbstractDataset):
        """Add a new dataset to the catalog.

        Args:
            name (str): name to identify the dataset
            dataset (AbstractDataset): the dataset object
        """
        self._logger.info(f"Adding dataset: {name}")
        self._datasets[name] = dataset


    def register(self,  name:str, dataset:AbstractDataset):
        """Register a new dataset with a given name.

        Args:
            name (str): name to identify the dataset
            dataset (AbstractDataset): the dataset object
        """
        self._logger.info(f"Adding dataset: {name}")
        self._datasets[name] = dataset
        

    def release(self, name:str):
        """Removes a dataset from the registered datasets.

        Args:
            name (str): name of the dataset to remove
        """
        self._logger.info(f"Removing dataset: {name}")
        del self._datasets[name]

    def __str__(self) -> str:
        return pprint.pformat(self.datasets)

    def __repr__(self) -> str:
        return pprint.pformat(self.datasets)

    def __eq__(self, other) -> bool:
        return all(
            [ds in other.datasets for ds in self.datasets] 
            + 
            [ds in self.datasets for ds in other.datasets] 
        )
        # return self.datasets == other.datasets
    