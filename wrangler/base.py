# Abstract Base Classes
from abc import ABC, abstractmethod
import logging
from loguru import logger
import inspect
from typing import Callable
from wrangler.utils import _get_imports
import itertools
import sys


class AbstractDataset(ABC):
    """[summary]
    """
    def __init__(self, name:str, decription:str=None) -> None:
        self.name = name
        self.description = decription

    @property
    def _logger(self):
        return logger

    def __str__(self):
        params = []
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, (str,int,float,list)):
                    params.append(f"{key}='{str(value)}'")
                else:
                    params.append(f"{key}={type(value).__name__}")
        params = ", ".join(params)  
        return f"{type(self).__name__}"+"("+params+")"

    def __repr__(self):
        params = []
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, (str,int,float,list)):
                    params.append(f"{key}='{str(value)}'")
                else:
                    params.append(f"{key}={type(value).__name__}")
        params = ", ".join(params)  
        return f"{type(self).__name__}"+"("+params+")"

    def load(self, *args, **kwargs):
        self._logger.info(f"Loading {str(self)}")
        return self._load(*args,**kwargs)

    def save(self, *args, **kwargs):
        self._logger.info(f"Saving {str(self)}" )
        return self._save(*args,**kwargs)
    

    def add_description(self, description):
        if self.description:
            self._logger.info(f"Replacing description of dataset {type(self).__name__}")    
        self.description = description
    

    @abstractmethod
    def _load(self, *args, **kwargs):
        pass

    @abstractmethod
    def _save(self, *args, **kwargs):
        pass



class AbstractTransformer(ABC):

    @property
    def _logger(self):
        return logger


    def fit(self, *args, **kwargs):
        """
        It fits the transformer. In some implementations
        this step could be empty, in other cases, this step
        could involve the computation of some information needed
        in the transformation step.

        Args:
            *args: all inputs needed for the transformer.
            **kwargs: all extra keyword arguments. 

        Returns:
            AbstractTransformer: the fitted transformer.
        """        
        # self._logger.debug(f"Fitting {type(self).__name__}")
        self._logger.info(f"Fitting {str(self)}")
        return self._fit(*args, **kwargs)


    def transform(self, *args, **kwargs):
        """Applies the transformation to the inputs and return the 
        corresponding outputs.

        Args:
            *args: all inputs needed for the transformer.
            **kwargs: all extra keyword arguments. 

        Returns:
            pd.DataFrame: the transformed dataframe.
        """        
        # self._logger.debug(f"Transforming {type(self).__name__}")
        self._logger.info(f"Transforming {str(self)}")
        return self._transform(*args, **kwargs)


    def fit_transform(self,*args, **kwargs):
        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)


    @abstractmethod
    def _fit(self, *args, **kwargs):
        pass


    @abstractmethod
    def _transform(self, *args, **kwargs):
        pass   


    def __str__(self):
        params = []
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, (str)):
                    params.append(f"{key}='{str(value)}'")
                elif isinstance(value, (int,float,list)):
                    params.append(f"{key}={str(value)}")
                elif callable(value):
                    params.append(f"{key}={str(value)}")
                else:
                    params.append(f"{key}={type(value).__name__}")
                
        params = ", ".join(params)  
        return f"{type(self).__name__}"+"("+params+")"


    def __repr__(self):
        params = []
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, (str)):
                    params.append(f"{key}='{str(value)}'")
                elif isinstance(value, (int,float,list)):
                    params.append(f"{key}={str(value)}")
                elif callable(value):
                    params.append(f"{key}={str(value)}")
                else:
                    params.append(f"{key}={type(value).__name__}")
        params = ", ".join(params)  
        return f"{type(self).__name__}"+"("+params+")"



class FunctionWrapper:

    """
    Wrapper for functions. It changes the ``__repr__``
    property with the name and the docstring of the function.
    in order to visualize it easier.

    Is also adds a ``__source__`` property which returns
    the source code of the function.

    Attributes:
        function (callable): ...
        fn_name (str): ...
    """    

    def __init__(self, function:Callable):
        self.function = function 
        # self.source = self._get_source(function)
    
    # def _get_source(self, function):
    #     import_execs = _get_imports(sys.modules[function.__module__])
    #     source = list(itertools.dropwhile(
    #         lambda line: line.startswith('@'), inspect.getsource(function).splitlines()
    #         ))
        
    #     for idx, imp in enumerate(import_execs):
    #         source.insert(1+idx, f"    {imp}")
    #     return source

    # def __call__(self, *args, **kwargs):
    #     exec('\n'.join(self.source))
    #     fn_name = self.function.__name__ 
    #     return eval(f"{fn_name}(*args, **kwargs)")

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


    def _fn_name(self, obj):
        return f"{obj.__name__}"
        # return f"{obj.__name__}: {obj.__doc__}"

    def __str__(self):
        # return inspect.getsource(self.function)
        return self._fn_name(self.function)
    
    def __repr__(self):
        return self._fn_name(self.function)


class AbstractDataCatalog(ABC):
    pass
