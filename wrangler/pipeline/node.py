from typing import Union,List,AnyStr,Dict, Callable,Any
import logging


class FunctionWrapper():
    pass

class Node(object):
    """_summary_

    Attributes:
        func (Callable): _description_
        inputs (Union[AnyStr,List]): _description_
        outputs (Union[AnyStr,List]): _description_
        func_kwargs (Union[Dict,None], optional): _description_. Defaults to None.
        name (AnyStr, optional): _description_. Defaults to None.
    """

    def __init__(
            self,
            func:Callable,
            inputs:Union[AnyStr,List],
            outputs:Union[AnyStr,List],
            func_kwargs:Union[Dict,None]=None,
            name:AnyStr=None
        ):

        self._name = name

        if callable(func):
            self._func = func
        else:
            msg = "transformer must be a callable`"
            self._logger.error(msg)
            raise ValueError(msg)

        self._inputs = [inputs] if isinstance(inputs, str) else inputs
        
        self._outputs = [outputs] if isinstance(outputs, str) else outputs

        self._func_kwargs = func_kwargs if func_kwargs else {}

    
    @property
    def _logger(self):
        return logging.getLogger(__name__)

    @property
    def inputs(self):
        """List of node inputs"""
        return self._inputs

    @property
    def func_kwargs(self):
        """Dictionary of node function key word arguments"""
        return self._func_kwargs

    @property
    def outputs(self):
        """List of node outputs"""
        return self._outputs

    @property
    def _defaut_name(self):
        return f"{self._func.__name__}"

    @property
    def name(self) -> str:
        """Node's name if provided or the name of its function"""
        node_name = f"{self._name if self._name is not None else self._defaut_name}"
        return node_name

    @property
    def _unique_key(self):
        def hashable(value):
            if isinstance(value, list):
                return tuple(value)
            return value

        return (self.name, hashable(self._inputs), hashable(self._outputs))

    @property
    def func(self):
        """Node's function to excecute."""
        return self._func
        
    def _format_outputs(self, outputs):
        if len(self._outputs)==1:
            outputs = {self._outputs[0]: outputs}
        else:
            outputs = dict(zip(self._outputs, outputs))
        return outputs     

    def run(self, inputs:Dict[str,Any])->Dict:
        """
        Excecute the node with the given input data.

        Args:
            inputs (Dict[str,Any]): the inputs in dictionary form to run the node

        Returns:
            Dict: node outputs
        """
        if not isinstance(inputs, dict):
            msg = "inputs must be passed as dictionary of {'name':data}"
            self._logger.error(msg)
            raise ValueError(msg)

        inputs = [inputs[item] for item in self._inputs]

        try:
            self._logger.error(f"running {self.name}")
            outputs_=self._func(*inputs, **self._func_kwargs)
        except Exception as e:
            self._logger.error(e)
            
        return self._format_outputs(outputs_)   


    def __str__(self):
        def _set_to_str(xset):
            return f"[{','.join(xset)}]"

        node_str = f"Node: {self._name if self._name else self._defaut_name}"

        in_str = _set_to_str(self._inputs) if self._inputs else "None"
        out_str = _set_to_str(self._outputs) if self._outputs else "None"
        
        tr_str = f"({in_str}) -> {out_str}\n"
        return node_str + tr_str


    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self._unique_key == other._unique_key

    def __lt__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self._unique_key < other._unique_key

    def __hash__(self):
        return hash(self._unique_key)
