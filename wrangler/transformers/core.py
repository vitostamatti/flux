
'''
Core Transformers are implementations for general purpose.


'''

from wrangler.base import AbstractTransformer, FunctionWrapper

import pandas as pd

from typing import Any, Tuple, Union, List, AnyStr, Dict, Callable

# class ColumnDoesNotExists(Exception):
    # pass

class ColumnNamesModifier(AbstractTransformer):
    """
    Changes the names of the columns.

    Attributes:
        columns (Union[AnyStr, List[AnyStr]], optional): [description]. Defaults to None.
        suffix (str, optional): [description]. Defaults to None.
        prefix (str, optional): [description]. Defaults to None.
    """ 
    def __init__(self, columns:Union[AnyStr, List[AnyStr]]=None, suffix:str=None, prefix:str=None):
       
        self.columns = columns
        self.suffix = suffix
        self.prefix = prefix    
  

    def _check_columns(self, X:pd.DataFrame):
        for col in self.columns:
            if col not in X.columns:
                raise Exception(f'Column {col} does not exists')

    def _fit(self, X:pd.DataFrame):       
        if isinstance(self.columns, str):
            self.columns = [self.columns]

        if not self.columns:
            self.columns = X.columns
        self._check_columns(X)
        return self

    def _transform(self, X:pd.DataFrame):
        new_columns = [f'{self.prefix}' + str(col) + f'{self.suffix}' for col in self.columns]
        X[self.columns].columns = new_columns
        return X


class ColumnSelector(AbstractTransformer):
    """
    Filters the input DataFrame to keep only the columns specified.

    Perform some validation regarding the columns to filter and
    the columns present in the input dataframe.

    Attributes:
        columns (Union[AnyStr, List[AnyStr]]): list of columns to select.

    """    
    def __init__(self, columns:Union[AnyStr, List[AnyStr]]):     
        self.columns = columns

    def _check_columns(self, X:pd.DataFrame):
        for col in self.columns:
            if col not in X.columns:
                raise Exception(f'Column {col} does not exists')

    def _fit(self, X:pd.DataFrame):  
        """
        Perform some validation regarding the columns to filter and
        the columns present in the input dataframe.

        Args:
            X (pd.DataFrame): the dataframe to fit on this transformer.

        Raises:
            Exception: When a column does not exists in the provided
                dataframe.

        Returns:
            ColumnSelector: returns the fitted object.
        """        
      
        if isinstance(self.columns, str):
            self.columns = [self.columns]
        self._check_columns(X)
        return self
    

    def _transform(self, X:pd.DataFrame):
        """
        Selects the columns from the given input and returns
        the filtered dataframe.

        Args:
            X (pd.DataFrame): the dataframe to transform on this transformer.

        Returns:
            pd.DataFrame: the dataframe with the columns selected.
        """        
        X = X[self.columns].copy()
        return X


class ColumnDropper(AbstractTransformer):
    """
    Drops the columns from the given input and returns the reduced dataframe.

    Perform some validation regarding the columns to drop and
    the columns present in the input dataframe.

    Attributes:
        columns (Union[AnyStr, List[AnyStr]]): list of columns to drop.
    """  
    
    def __init__(self, columns:Union[AnyStr, List[AnyStr]]):      
        self.columns = columns

    def _check_columns(self, X:pd.DataFrame):
        for col in self.columns:
            if col not in X.columns:
                raise Exception(f'Column {col} does not exists')

    def _fit(self, X:pd.DataFrame):
        """
        Perform some validation regarding the columns to drop and
        the columns present in the input dataframe.

        Args:
            X (pd.DataFrame): the dataframe to fit on this transformer.

        Raises:
            Exception: When a column does not exists in the provided
                dataframe.

        Returns:
            ColumnDropper: returns the fitted object
        """  
        if isinstance(self.columns, str):
            self.columns = [self.columns]

        self._check_columns(X)
        return self 

    def _transform(self, X:pd.DataFrame):
        """Drops the columns from the given input and returns
        the reduced dataframe.

        Args:
            X (pd.DataFrame): the dataframe to transform on this transformer.

        Returns:
            pd.DataFrame: the dataframe with the columns dropped.
        """    
        X = X.loc[:,~(X.columns.isin(self.columns))].copy()
        return X


class ColumnRenamer(AbstractTransformer):
    """
    Rename the given key columns by its corresponding value.

    Perform some validation regarding the columns to rename and
    the columns present in the input dataframe.

    Attributes:
        columns (Dict[str,str]): dictionary of actual column 
            names and new names.
    """  
    
    def __init__(self, columns:Dict[str,str]):   
        self.columns = columns

    def _fit(self, X:pd.DataFrame, y:pd.DataFrame=None):
        """
        [Summary]

        Args:
            X (pd.DataFrame): [description]
            y (pd.DataFrame, optional): [description]. Defaults to None.

        Returns:
            ColumnRenamer: returns the fitted object
        """        
        for col in self.columns:
            if col not in X.columns:
                raise Exception(f'Column {col} does not exists')

        return self 

    def _transform(self, X:pd.DataFrame):
        """[summary]

        Args:
            X (pd.DataFrame): [description]

        Returns:
            pd.DataFrame: [description]
        """   
        X = X.copy()

        X.rename(columns=self.columns, inplace=True)

        return X


class ChangeDataType(AbstractTransformer):
    """Changes the data type of the given columns
    to the dtype specified.

    Attributes:
        columns (list): list of column names to apply 
            transformation.
        dtype : a python compatible data type.
    """    

    def __init__(self, columns:list, dtype):
     
        self.columns = columns  
        self.dtype = dtype

    def _fit(self, X:pd.DataFrame):  
        """
        [Summary]

        Args:
            X (pd.DataFrame): [description]

        Returns:
            ColumnSelector: returns the fitted object
        """        
        X = X.copy()
        try:
            X[self.columns].astype(self.dtype)
        except:
            raise ValueError(f'Columns {self.columns} cannot be converted to {self.dtype}')

        return self
    
    def _transform(self, X:pd.DataFrame):
        """[summary]

        Args:
            X (pd.DataFrame): [description]

        Returns:
            [type]: [description]
        """        
        X = X.copy()
        X[self.columns] = X[self.columns].astype(self.dtype)

        return X



class DataCheckPoint(AbstractTransformer):
    def __init__(self, save_path:str):
        self.save_path = save_path

    def _fit(self, X:pd.DataFrame):
        return self 

    def _transform(self, X:pd.DataFrame):
        X = X.copy()
        X.to_csv(self.save_path)
        return X



class DataTypeCheck(AbstractTransformer):

    def _fit(self, X:pd.DataFrame):
        self.data_types = X.dtypes.to_dict()
        return self 

    def _transform(self, X:pd.DataFrame):
        X = X.copy()
        for col_name, typ in self.data_types.items():
            if col_name not in X.columns:
                raise ValueError(f"Column {col_name} not in fitted DataFrame")
            if (typ != X[col_name].dtype):
                raise ValueError(f"Column {col_name} was fitted with datatype {typ} instead of {X[col_name].dtype}.")

        return X


class GroupbyTransformer(AbstractTransformer):
    """Basic groupby implementacion using the DataFrame.groupby(by=)[subcols].agg({})
    method.

    Attributes:
        by (Union[str,List[str]]): The column or list of columns to groupby
        agg (Dict[str,Any]): The columns and functions to aggregate them
        subcols (List[str], optional): The subset columns to select and apply aggregations. Defaults to None.
            If None, all columns are selected.
    """     
    def __init__(self, by:Union[str,List[str]], agg:Dict[str,Any], subcols:List[str]=None) -> None:
       
        self.by = by
        self.agg = agg
        self.subcols = subcols

    def _fit(self, X:pd.DataFrame):
        pass

    def _transform(self, X:pd.DataFrame):
        X = X.copy()

        if self.subcols is not None:
            X = X.groupby(self.by)[self.subcols].agg(self.agg)
        else:
            X = X.groupby(self.by).agg(self.agg)
        return X


class PivotTable(AbstractTransformer):
    """
    Reshape data to produce a “pivot” table based on column values

    If an aggfunc is not passed, it performs a Dataframe.pivot() method.
    If an aggfunct is passed, if performs a pd.pivot_table() method with all its keyword arguments.

    Attributes:
        index (Union[str,List[str]]): Keys to group by on the pivot table index.
        columns (List[str]): Keys to group by on the pivot table column
        values (List[str], optional): column to aggregate
        aggfunc (str, optional): aggregation function to apply if required. Defaults to None.
        pivot_table_kwargs: all others kwargs of the pd.pivot_table() method.

    """
    def __init__(self, index:Union[str,List[str]], columns:List[str], values:List[str]=None, aggfunc:str=None, **pivot_table_kwargs): 
        self.index  = index
        self.columns = columns
        self.values = values
        self.aggfunc = aggfunc
        self.pivot_table_kwargs = pivot_table_kwargs
        

    def _do_pivote(self, X:pd.DataFrame):
        if self.aggfunc == None:
            X = X.pivot(index=self.index, columns=self.columns, 
                values=self.values)
        else:
            X = pd.pivot_table(X, index=self.index, 
                columns=self.columns, values=self.values, 
                aggfunc=self.aggfunc,
                **self.pivot_table_kwargs)
        return X

    def _fit(self, X:pd.DataFrame):
        X = X.copy()
        X = self._do_pivote(X)
        self.pivot_columns = list(X.columns)
        return self 

    def _transform(self, X:pd.DataFrame):
        X = X.copy()
        X = self._do_pivote(X)
        X = pd.DataFrame(data = X.values,columns=self.pivot_columns,
            index = X.index)
        return X



class JoinTransformer(AbstractTransformer):
    """
    Perform the DataFrame.merge method with the two inputs 

    First input is the left and second input is the right.

    Attributes:
        how (str, optional): the type of merge to perform (inner, outer, left, right)
        on (str): Column or index level names to join on
        left_on (str): columns of the left input to join
        right_on (str): columns of the right input to join
        left_index (bool, optional): Defaults to False.
        right_index (bool, optional):  Defaults to False.
        suffixes (Union[List[str],Tuple[str]], optional):  Defaults to ('_x', '_y').
    """
    def __init__(self, how='inner', on:str=None, left_on:Union[str,List[str]]=None, 
            right_on:Union[str,List[str]]=None, left_index:bool=False, 
            right_index:bool=False, suffixes:Union[List[str],Tuple[str]]=('_x', '_y'), **kwargs) -> None:
           
        self.how = how
        self.on = on
        self.left_on = left_on
        self.right_on = right_on
        self.left_index = left_index
        self.right_index = right_index
        self.suffixes=suffixes
        self.merge_kwargs = kwargs
    
    def _fit(self, left:pd.DataFrame, right:pd.DataFrame):
        '''
        Chequeo de columnas right/left
        Chequeo de tipo de dato?
        '''
        return self

    def _transform(self, left, right):
        # left = inputs[0]
        # right = inputs[1]
        X = left.merge(right, 
            how=self.how, on=self.on, left_on=self.left_on, 
            right_on=self.right_on, left_index=self.left_index, 
            right_index=self.right_index, suffixes=self.suffixes,
            **self.merge_kwargs
        )
        # X.drop(columns=self.right_on,inplace=True)
        return X



class ColumnTransformer(AbstractTransformer):
    """
    Applies a function to a column of the input dataframe. 

    All parameters of the function must be passed as a dictionary in the func_kwargs
    attribute.

    Attributes:
        column (str): ...
        function (callable): ...
        func_kwargs (dict, optional): ... Defaults to None.
    """  

    def __init__(self, column:AnyStr, function:Callable, func_kwargs:Dict[AnyStr,Any]=None):    
        self.column = column
        self.function = FunctionWrapper(function)
        # self.function = function
        self.func_kwargs = func_kwargs
        

    def _fit(self, X:pd.DataFrame):
        """
        [Summary]

        Args:
            X (pd.DataFrame): [description]

        Returns:
            ColumnTransformer: returns the fitted object
        """ 
        if self.column not in X.columns:
            raise Exception("Column {col} does not exists")
        return self
        
    def _transform(self, X:pd.DataFrame):
        """[summary]

        Args:
            X (pd.DataFrame): [description]

        Returns:
            pd.DataFrame: [description]
        """    
        X = X.copy()
        
        if self.func_kwargs:
            X[self.column] = X[self.column].apply(lambda x: self.function(x,**self.func_kwargs))
        else:
            X[self.column] = X[self.column].apply(lambda x: self.function(x))
        
        return X



class DataframeTransformer(AbstractTransformer):
    """
    Applies a function to its inputs and return outputs.
    It does not have a specific `fit` method

    Inputs must be a Pandas DataFrame.

    All parameters of the function must be passed in the func_kwargs attribute. 

    Attributes:
        function (Callable): [description]
        func_kwargs (Dict[AnyStr,Any], optional): [description]. Defaults to None.
    """    
    def __init__(self, function:Callable, func_kwargs:Dict[AnyStr,Any]=None):
     
        # self.function = function
        self.function = FunctionWrapper(function)
        self.func_kwargs = func_kwargs

    def _fit(self, *inputs):
        return self
        
    def _transform(self, *inputs):
        if self.func_kwargs:
            outputs = self.function(*inputs, **self.func_kwargs)
        else:
            outputs = self.function(*inputs)
        return outputs
