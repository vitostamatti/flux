
from typing import Union,List
from wrangler.pipeline.node import Node
from collections import Counter
from itertools import chain
import logging
from loguru import logger

class Pipeline():
    """[summary]

    Attributes:
        nodes (List[Node], optional): the list of the nodes in the Pipeline. Defaults to None.
            The order of intertion will be the order of excecution.

    Example:

        .. code-block:: python

            pipeline = Pipeline()
            node = Node(
                name='first_node',
                transformer = ColumnDropper(column='useless_col'), 
                inputs = 'cars',
                outputs = 'cars_output'
            ) 
            pipeline.add(node)
            
    """    
    def __init__(self, nodes:List[Node]=None) -> None:
        self.nodes = nodes if nodes is not None else []
        
    @property
    def _logger(self):
        return logger

    def transform(self, data_catalog):
        self._run_pipeline(data_catalog, run_mode ='transform')

    def fit_transform(self, data_catalog):
        self._run_pipeline(data_catalog, run_mode ='fit_transform')
        
    def _run_pipeline(self, data_catalog, run_mode='fit_transform'):
        load_counts = Counter(chain.from_iterable(n.inputs for n in self.nodes))
        save_counts = Counter(chain.from_iterable(n.outputs for n in self.nodes))

        for node in self.nodes:
            if run_mode == 'fit_transform':
                node.fit_transform(data_catalog)

            elif run_mode == 'transform':
                node.transform(data_catalog)

            for data_set in node.inputs:
                load_counts[data_set] -= 1
                if (load_counts[data_set] < 1) and (data_set not in self.inputs()) and (save_counts.get(data_set, 0)==0):
                    data_catalog.release(data_set)

    def _exists(self, new_node):
        for node in self.nodes:
            if node.name == new_node.name:
                return True
    
    def _node_idx(self, node):
        for idx, self_node in enumerate(self.nodes):
            if node.name == self_node.name:
                return idx         

    def add(self, node:Union[Node,List[Node]]):
        if isinstance(node, Node):
            node = [node] 
        for n in node:
            if self._exists(n):
                self._logger.warning(f"Replacing Existing Node: {n.name}")
                pos = self._node_idx(n)
                self.nodes[pos] = n
            else:
                self.nodes.append(n)
                        
            self._logger.info(f"Node {n.name} added to Pipeline ")

    def __repr__(self):
        nodes = [f'- {str(node)}' for node in self.nodes]
        nodes = "".join(node for node in nodes)
        return f'Pipeline \n {nodes}'
    
    def __str__(self):
        nodes = [f'- {str(node)}' for node in self.nodes]
        nodes = "".join(node for node in nodes)
        return f'Pipeline \n {nodes}'

    def all_inputs(self):
        return set.union(set(), *(node.inputs for node in self.nodes))

    def all_outputs(self):
        return set.union(set(), *(node.outputs for node in self.nodes))

    def intermediates_datasets(self):
        return set.intersection(self.all_inputs(), self.all_outputs())

    def inputs(self):
        return set.difference(self.all_inputs(), self.intermediates_datasets())
    
    def outputs(self):
        return set.difference(self.all_outputs(), self.intermediates_datasets())

    def last_output(self):
        outputs = self.nodes[-1].outputs
        if isinstance(outputs, list):
            return outputs[-1]
        else:
            return outputs

    # def release_datasets(self):
    #     pass
    