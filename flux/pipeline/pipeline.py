
from typing import Union,List
from flux.pipeline.node import Node
from collections import defaultdict
import logging
from functools import reduce as _reduce


class Pipeline(object):
    """
    Allocates a list of `Node` elements.

    Attributes:
        nodes (List[Node], optional): the list of the nodes in the Pipeline. Defaults to None.
            The order of intertion will be the order of excecution.

    Example:

        .. code-block:: python

            pipe = Pipeline()
    """    

    def __init__(self, nodes:List[Node]=None) -> None:
        self._nodes = nodes if nodes is not None else []


    @property
    def _nodes_by_name(self):
        """Nodes in dictionary format with its name as key"""
        return {node.name: node for node in self._nodes}


    @property
    def _nodes_by_input(self):
        """Nodes in dictionary format with its inputs as key"""
        _nodes_by_input = defaultdict(set) 
        for node in self._nodes:
            for input_ in node.inputs:
                _nodes_by_input[input_].add(node)
        return _nodes_by_input


    @property
    def _nodes_by_output(self):
        """Nodes in dictionary format with its outputs as key"""
        _nodes_by_output = {}
        for node in self._nodes:
            for output in node.outputs:
                _nodes_by_output[output] = node
        return _nodes_by_output


    @property
    def _topo_sorted_nodes(self):
        """List of sorted nodes"""
        return list(toposort(self.node_dependencies))


    @property
    def _logger(self):
        return logging.getLogger(__name__)


    @property
    def node_dependencies(self):
        """All dependencies of nodes where the first Node has a direct dependency on
        the second Node.

        Returns:
            Dictionary where keys are nodes and values are sets made up of
            their parent nodes. Independent nodes have this as empty sets.
        """
        dependencies = {
            node: set() for node in self._nodes
        }
        for parent in self._nodes:
            for output in parent.outputs:
                for child in self._nodes_by_input[output]:
                    dependencies[child].add(parent)

        return dependencies


    @property
    def all_inputs(self):
        "All the unique nodes inputs"
        return set.union(set(), *(node.inputs for node in self.nodes))
    

    @property
    def all_outputs(self):
        "All the unique nodes outputs"
        return set.union(set(), *(node.outputs for node in self.nodes))


    @property
    def datasets(self):
        """The names of all data sets used by the ``Pipeline``,
        including inputs and outputs.
        Returns:
            The set of all pipeline data sets.
        """
        return set.union(self.all_inputs, self.all_outputs)


    @property
    def intermediates_inputs(self):
        """Intermediate inputs of the pipeline"""
        return set.intersection(self.all_inputs, self.all_outputs)


    @property
    def inputs(self):
        """Required inputs of the pipeline."""
        return set.difference(self.all_inputs, self.intermediates_inputs)
    

    @property
    def outputs(self):
        """Final outputs of the pipeline."""
        return set.difference(self.all_outputs, self.intermediates_inputs)


    @property
    def last_output(self):
        outputs = self.nodes[-1].outputs
        if isinstance(outputs, list):
            return outputs[-1]
        else:
            return outputs


    @property
    def nodes(self):
        """List of nodes in the pipeline"""
        return self._topo_sorted_nodes


    def _exists(self, new_node:Node):
        """Check if a node is already in the pipeline"""
        for node in self._nodes:
            if node.name == new_node.name:
                return True
    

    def _node_idx(self, node:Node):
        """Get the node index in the pipeline"""
        for idx, self_node in enumerate(self._nodes):
            if node.name == self_node.name:
                return idx         


    def add_nodes(self, node:Union[Node,List[Node]]):
        """Add a node or list of nodes to the pipeline

        Args:
            node (Union[Node,List[Node]]): node or list of nodes
        """
        if isinstance(node, Node):
            node = [node] 
        for n in node:
            if self._exists(n):
                self._logger.warning(f"Replacing Existing Node: {n.name}")
                pos = self._node_idx(n)
                self._nodes[pos] = n
            else:
                self._nodes.append(n)
                        
            self._logger.info(f"Node {n.name} added to Pipeline ")


    def __repr__(self):
        nodes = [f'- {str(node)}' for node in self.nodes]
        nodes = "".join(node for node in nodes)
        return f'Pipeline \n {nodes}'
    

    def __str__(self):
        nodes = [f'- {str(node)}' for node in self.nodes]
        nodes = "".join(node for node in nodes)
        return f'Pipeline \n {nodes}'


    def __add__(self, other):
        if not isinstance(other, Pipeline):
            return NotImplemented
        return Pipeline(set(self.nodes + other.nodes))


    def __eq__(self, other):
        return all(
            [n in other.nodes for n in self.nodes]
            +
            [n in self.nodes for n in other.nodes]
            )

        

def toposort(data):
    def _toposort(data):
        """Dependencies are expressed as a dictionary whose keys are items
        and whose values are a set of dependent items. Output is a list of
        sets in topological order. The first set consists of items with no
        dependences, each subsequent set consists of items that depend upon
        items in the preceding sets.
        """

        # Special case empty input.
        if len(data) == 0:
            return None

        # Ignore self dependencies.
        for k, v in data.items():
            v.discard(k)

        # Find all items that don't depend on anything.
        extra_items_in_deps = _reduce(set.union, data.values()) - set(data.keys())
        # Add empty dependences where needed.
        data.update({item:set() for item in extra_items_in_deps})
        while True:

            ordered = sorted(set(item for item, dep in data.items() if len(dep) == 0))
            if not ordered:
                break

            for item in ordered:
                yield item
                data.pop(item, None)

            for dep in sorted(data.values()):
                dep -= set(ordered)

        if len(data) != 0:
            msg = 'Cyclic dependencies exist among these items: {}'
            raise ValueError(msg.format(' -> '.join(repr(x) for x in data.keys())))

    return list(_toposort(data))



