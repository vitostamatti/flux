
from typing import Union, List, Any

from wrangler.base import AbstractTransformer

from wrangler.data.catalog import DataCatalog

from loguru import logger


class Node():
    """[summary]

    Attributes:
        name (str): identifier name of the node
        transformer (AbstractTransformer): the transformer to run on this node.
        inputs (Union[str,List[str]]): the input or list of inputs to pass to the transformer.
            This depends on the specific parameters of the fit and transform methods of the transformer.
        outputs (Union[str,List[str]]): the output or list of outputs that returns from the transformer.

    Example:

        .. code-block:: python

            node = Node(
                name='first_node',
                transformer = ColumnDropper(column='useless_col'), 
                inputs = 'cars',
                outputs = 'cars_output'
            ) 

    """    
    def __init__(self, name:str, transformer:AbstractTransformer, 
        inputs:Union[str,List[str]], outputs:Union[str,List[str]]
        ) -> None:
        
        self.name = name
        self.transformer = transformer

        self.inputs = [inputs] if isinstance(inputs, str) else inputs
        self.outputs = [outputs] if isinstance(outputs, str) else outputs
        
        
    def __str__(self):
        return ('name={name}\n'
        'transformer= {transformer}\n'
        'inputs= {inputs}\n'
        'outputs= {outputs}\n'
        ).format(**self.__dict__)


    def __repr__(self):
        return ('name={name}\n'
        'transformer= {transformer}\n'
        'inputs= {inputs}\n'
        'outputs= {outputs}\n'
        ).format(**self.__dict__)

    @property
    def _logger(self):
        # return logging.getLogger(self.__module__)
        # return logging.getLogger((type(self).__name__))
        # return logging.getLogger(LOGGER_NAME)
        return logger

    def fit(self, data_catalog:DataCatalog):
        if len(self.inputs)==1:
            inputs = data_catalog.load(self.inputs)
        else: 
            inputs = []
            for input in self.inputs:
                inputs.append(data_catalog.load(input))
                
        self.transformer.fit(inputs)
        
        return self

    def fit_transform(self, data_catalog:DataCatalog):
        outputs = self._run_node(data_catalog, fit=True)
        return outputs

    def transform(self,data_catalog:DataCatalog):
        outputs = self._run_node(data_catalog, fit=False)
        return outputs

    def _run_node(self, data_catalog, fit:bool=True):
        self._logger.info(f"Running Node: {self.name}")
 
        inputs = self._load_inputs(data_catalog)

        if fit:
            self.transformer.fit(*inputs)
        
        outputs = self.transformer.transform(*inputs)

        self._save_outputs(outputs, data_catalog)

        return outputs 

    def _load_inputs(self, data_catalog:DataCatalog):
        inputs = []
        if len(self.inputs)==1:
            input = data_catalog.load(self.inputs[0])
            inputs.append(input)
        else:
            for input in self.inputs:
                inputs.append(data_catalog.load(input))
        return inputs
    
    def _save_outputs(self, outputs:Any, data_catalog:DataCatalog):
        if len(self.outputs)==1:
            data_catalog.save(self.outputs[0], outputs)
        else:
            for idx, output in enumerate(self.outputs):
                data_catalog.save(output, outputs[idx])