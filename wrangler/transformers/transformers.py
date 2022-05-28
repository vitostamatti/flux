"""
Ejemplo de docstring

"""

import numpy as np
from pandas.core.frame import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.pipeline import FeatureUnion, Pipeline, _fit_transform_one, _transform_one
from scipy import sparse

import inspect

from typing import List, NewType, Union, List

from abc import ABC

class Transformer():
    def __init__(self, *args, **kwargs) -> None:
        pass

    def fit(self, *args, **kwargs):
        pass

    def transform(self, *args, **kwargs):
        pass

    def __str__(self):
        params = []
        for key, value in self.__dict__.items():
            if isinstance(value, (str,int,float,list)):
                params.append(f"{key}={str(value)}")
            else:
                params.append(f"{key}={type(value).__name__}")
        params = ", ".join(params)  
        return f"{type(self).__name__}"+"("+params+")"

    def __repr__(self):
        params = []
        for key, value in self.__dict__.items():
            if isinstance(value, (str,int,float,list)):
                params.append(f"{key}={str(value)}")
            else:
                params.append(f"{key}={type(value).__name__}")
        params = ", ".join(params)  
        return f"{type(self).__name__}"+"("+params+")"


class MultiOutputTransformer(Transformer):
    def __init__(self, column:str) -> None:
        self.column = column
    
    def fit(self, inputs):
        self.aux_output = inputs[self.column]
        return self

    def transform(self, inputs):
        return inputs, self.aux_output


class JoinTransformer(Transformer):
    def __init__(self, left_on, right_on, how) -> None:
        self.left_on = left_on
        self.right_on = right_on
        self.how = how
    
    def fit(self, inputs):
        '''
        Chequeo de columnas right/left
        Chequeo de tipo de dato?
        '''
        return self

    def transform(self, inputs):
        X = inputs[0].merge(
            inputs[1], 
            left_on = self.left_on,
            right_on = self.right_on,
            how = self.how
        )
        return X


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

    def __init__(self, function:callable):
        self.function = function 
        self.__source__ = str(inspect.getsource(self.function))

    def _fn_name(self, obj):
        return f"<{obj.__name__}: {obj.__doc__}>"

    def __call__(self,*args,**kwargs): 
        return self.function(*args,**kwargs)
    
    def __str__(self):
        return inspect.getsource(self.function)
    
    def __repr__(self):
        return self._fn_name(self.function)


class ChangeDataType(TransformerMixin):
    def __init__(self, columns:list, dtype:str):
        self.columns = columns  
        self.dtype = dtype

    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None):  
        """
        [Summary]

        Args:
            X (pd.DataFrame): [description]
            y (pd.DataFrame, optional): [description]. Defaults to None.

        Returns:
            ColumnSelector: returns the fitted object
        """        
        X = X.copy()
        try:
            X[self.columns].astype(self.dtype)
        except:
            raise ValueError(f'Columns {self.columns} cannot be converted to {self.dtype}')

        return self
    
    def transform(self, X:pd.DataFrame):
        """[summary]

        Args:
            X (pd.DataFrame): [description]

        Returns:
            [type]: [description]
        """        
        X = X.copy()
        X[self.columns] = X[self.columns].astype(self.dtype)

        return X


class ColumnSelector(TransformerMixin, BaseEstimator):
    """
    It filters the input DataFrame to keep only the
    columns specified.

    It follows the Scikit-Learn API and has a fit and transform methods.

    Attributes:
        columns (list[str]): ...

    """    
    def __init__(self, columns:list):
        self.columns = columns

    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None):  
        """
        [Summary]

        Args:
            X (pd.DataFrame): [description]
            y (pd.DataFrame, optional): [description]. Defaults to None.

        Returns:
            ColumnSelector: returns the fitted object
        """        
        return self
    
    def transform(self, X:pd.DataFrame):
        """[summary]

        Args:
            X (pd.DataFrame): [description]

        Returns:
            [type]: [description]
        """        
        X = X[self.columns].copy()
        return X

 
class ColumnDropper(TransformerMixin, BaseEstimator):
    """
    [summary]}

    Attributes:
        columns (list[str]): ...
    """  
    
    def __init__(self, columns:list):
        self.columns = columns

    def transform(self, X:pd.DataFrame):
        """[summary]

        Args:
            X (pd.DataFrame): [description]

        Returns:
            [type]: [description]
        """    
        X = X.loc[:,~(X.columns.isin(self.columns))].copy()
        return X

    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None):
        """
        [Summary]

        Args:
            X (pd.DataFrame): [description]
            y (pd.DataFrame, optional): [description]. Defaults to None.

        Returns:
            ColumnDropper: returns the fitted object
        """   
        return self 


class ColumnRenamer(TransformerMixin, BaseEstimator):
    """
    [summary]}

    Attributes:
        columns (list[str]): ...
    """  
    
    def __init__(self, columns:dict):
        self.columns = columns

    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None):
        """
        [Summary]

        Args:
            X (pd.DataFrame): [description]
            y (pd.DataFrame, optional): [description]. Defaults to None.

        Returns:
            ColumnRenamer: returns the fitted object
        """        
        return self 

    def transform(self, X:pd.DataFrame):
        """[summary]

        Args:
            X (pd.DataFrame): [description]

        Returns:
            pd.DataFrame: [description]
        """   
        X = X.copy()

        X.rename(columns=self.columns, inplace=True)

        return X


class ColumnTextConcatenator(TransformerMixin,BaseEstimator):
    """
    [summary]}

    Attributes:
        columns (list[str]): ...
        operator (str): ...
        drop_original (bool): ...
    """  
    def __init__(self, columns:List[str], operator:str="_", drop_original:bool = False):
        self.columns = columns
        self.operator = operator
        self.drop_original=drop_original
        
    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None):
        """
        [Summary]

        Args:
            X (pd.DataFrame): [description]
            y (pd.DataFrame, optional): [description]. Defaults to None.

        Returns:
            ColumnTextConcatenator: returns the fitted object
        """ 
        return self
        
    def transform(self, X:pd.DataFrame,**kwargs):
        """[summary]

        Args:
            X (pd.DataFrame): [description]

        Returns:
            pd.DataFrame: [description]
        """          
        X = X.copy()
        
        new_col_name = "concat_" + self.operator.join(self.columns)
        
        X[new_col_name] = X[self.columns].agg(self.operator.join, axis=1)
        
        if self.drop_original:
            X.drop(columns=self.columns, inplace=True)

        return X  


class ColumnTransformer(TransformerMixin,BaseEstimator):
    """
    [summary]}

    Attributes:
        column (str): ...
        function (callable): ...
        func_kwargs (dict, optional): ... Defaults to None.
    """  

    def __init__(self, column:str, function:callable, func_kwargs:dict=None):    
        self.column = column
        self.function = FunctionWrapper(function)
        self.func_kwargs = func_kwargs
        
    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None):
        """
        [Summary]

        Args:
            X (pd.DataFrame): [description]
            y (pd.DataFrame, optional): [description]. Defaults to None.

        Returns:
            ColumnTransformer: returns the fitted object
        """ 
        return self
        
    def transform(self, X:pd.DataFrame):
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
    
    
class DataframeTransformer(TransformerMixin,BaseEstimator):

    def __init__(self, function, func_kwargs=None):
        self.function = FunctionWrapper(function)
        self.func_kwargs = func_kwargs

    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None):
        return self
        
    def transform(self, X:pd.DataFrame):
        X = self.function(X.copy())
        return X


# Numeric
class ClipTransformer(TransformerMixin,BaseEstimator):
    def __init__(self,column:str, min_value:float, max_value:float):
        self.column = column
        self.min_value = min_value
        self.max_value = max_value
    
    def _clip_func(self, x):
        if x < self.min_value:
            return self.min_value
        elif x > self.max_value:
            return self.max_value
        else:
            return x       
    
    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None):
        return self
        
    def transform(self, X:pd.DataFrame):
        X = X.copy()
        X[self.column] = X[self.column].apply(lambda x: self._clip_func(x))
        return X
    
# Numeric
class OutliersQuantileTransformer(TransformerMixin, BaseEstimator):
    def __init__(self,column:str, q_low:float, q_up:float, drop=False):
        self.column = column
        self.q_low = q_low
        self.q_up = q_up
        self.drop = drop
        
        
    def _clip_func(self, x):
        if x < self.q_low_value:
            return np.NaN
        elif x > self.q_up_value:
            return np.NaN
        else:
            return x  
    
    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None):
        X = X.copy()
        self.q_low_value = X[self.column].quantile(self.q_low)
        self.q_up_value = X[self.column].quantile(self.q_up)
        return self
        
    def transform(self, X:pd.DataFrame):
        
        check_is_fitted(self, 'q_low_value')
        check_is_fitted(self, 'q_up_value')
        
        X = X.copy()
        X[self.column] = X[self.column].apply(lambda x: self._clip_func(x))
        if self.drop:
            X = X[~X[self.column].isnull()]
        return X


# Text
class TextConverter(TransformerMixin, BaseEstimator):
    def __init__(self, columns:list):
        self.columns = columns

    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None, **kwargs):
        return self
    
    def transform(self, X:pd.DataFrame, **kwargs):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].astype(str)
        return X
    

# Text   
class TextFillNa(TransformerMixin, BaseEstimator):
    def __init__(self, columns:list, value='sin_dato'):
        self.columns = columns
        self.value = value

    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None, **kwargs):
        return self
    
    def transform(self, X:pd.DataFrame, **kwargs):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].fillna(self.value)
        return X    
    

# Text
class TextKeepTopNCategories(TransformerMixin, BaseEstimator):
    def __init__(self, column:str, max_cats:int, left_value:str = 'otra'):
        self.column = column
        self.max_cats = max_cats
        self.left_value = left_value

    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None, **kwargs):
        cats = X[self.column].value_counts().index
        
        if len(cats) > self.max_cats:
            self.categories_ = cats[:self.max_cats]
        else:
            self.categories_ = cats
        
        return self
    
    def transform(self, X:pd.DataFrame, **kwargs):
        
        check_is_fitted(self, 'categories_')
        X = X.copy()
        X[self.column] = X[self.columns].apply(lambda x: x if x in self.categories_ else self.left_value)
        
        return X

# Text    
class TextSplitByCharacter(TransformerMixin, BaseEstimator):
    def __init__(self, column:str, split_string:str = " "):
        self.column = column
        self.split_string = split_string
    
    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None, **kwargs):
        return self
    
    def transform(self, X:pd.DataFrame, **kwargs):
        X=X.copy()
        X = X[self.column].str.split(self.split_string, expand=True)
        X.columns = [f'{self.column}_splited_{i}' for i in X.columns]
        
        return X


# Text
class ModeImputer(TransformerMixin, BaseEstimator):
    
    def __init__(self, column:str):
        self.column = column       
    
    
    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None):
        try:
            X = X.copy()
            self.impute_value = X[self.column].mode()[0]
        except Exception as e:
            print(e)
        finally:
            return self
            
            
    def transform(self, X:pd.DataFrame, y:pd.DataFrame=None):
    
        check_is_fitted(self, 'impute_value')
        X = X.copy()
        X[self.column] = X[self.column].fillna(self.impute_value)
        
        return X

# Text        
class GroupModeImputer(TransformerMixin):
    
    def __init__(self, feature_name:str, group_cols:list):
        self.feature_name = feature_name   
        self.group_cols = group_cols
    
    
    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None):
        
        try:
            X = X.copy()
            self.impute_map = X.groupby(self.group_cols)[self.feature_name].agg(
                lambda x:x.value_counts().index[0]
            ).reset_index(drop=False)
            
        except Exception as e:
            print(e)
        finally:
            return self
            
    def transform(self, X:pd.DataFrame, y:pd.DataFrame=None):
        
        # make sure that the imputer was fitted
        check_is_fitted(self, 'impute_map') 
        X = X.copy()
        for _, row in self.impute_map.iterrows():
            ind = (X[self.group_cols] == row[self.group_cols]).all(axis=1)
            X.loc[ind, self.feature_name] = X.loc[ind, self.feature_name].fillna(row[self.feature_name])
        
        return X
    
# Numeric 
class GroupImputer(BaseEstimator, TransformerMixin):
    """Class used for imputing missing values in a 
    pd.DataFrame using either mean or median of a group.

    Args:
        group_cols (list): List of columns used for calculating the aggregated value 
        target (str): The name of the column to impute
        metric (str, optional): The metric to be used for remplacement, 
            can be one of ['mean', 'median']. Defaults to 'mean'.

    Raises:
        ValueError: [description]
    """        

    def __init__(self, group_cols:list, target:str, metric='mean'):
 
        if metric not in ['mean','median']:
            raise ValueError('Unrecognized value for metric, should be mean/median')
        
        self.group_cols = group_cols
        self.target = target
        self.metric = metric
    
    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None):
        
        impute_map = X.groupby(self.group_cols)[self.target].agg(
            self.metric
        ).reset_index(drop=False)

        self.impute_map = impute_map
        return self 
    
    def transform(self, X:pd.DataFrame, y:pd.DataFrame=None):
        
        # make sure that the imputer was fitted
        check_is_fitted(self, 'impute_map')
        
        X = X.copy()
        
        for index, row in self.impute_map.iterrows():
            ind = (X[self.group_cols] == row[self.group_cols]).all(axis=1)
            X.loc[ind, self.target] = X.loc[ind, self.target].fillna(row[self.target])
        
        return X


############### Deprecated ###############
class LookUpTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, lookup_table:pd.DataFrame, left_cols:list, right_cols:list):
        self.lookup_table = lookup_table
        self.left_cols = left_cols
        self.right_cols = right_cols
    
    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None, **kwargs):
        self.lookup_table_ = self.lookup_table.rename(columns = {col:f"_{col}" for col in self.right_cols})
        self.right_cols_ = [f"_{col}" for col in self.right_cols]
        return self
    
    def transform(self, X:pd.DataFrame, **kwargs):
        X=X.copy()
        
        X = X.merge(self.lookup_table_, left_on=self.left_cols, right_on=self.right_cols_, how='left')
    
        X.drop(columns=self.right_cols_, inplace=True)

        return X


############### Deprecated ###############
class PandasFeatureUnion(FeatureUnion):
    
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans,
                X=X,
                y=y,
                weight=weight,
                **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)

    def transform(self, X:pd.DataFrame):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs


# Text
# import category_encoders as ce
# class OneHotTransformer(TransformerMixin):
    
#     def __init__(self, column:str):
#         self.column = column

#     def fit(self, X:pd.DataFrame, y:pd.DataFrame=None):
#         X = X.copy()

#         self.encoder = ce.OneHotEncoder( 
#             cols=self.column, 
#             return_df=True, 
#             handle_missing='value', 
#             handle_unknown="value",
#             use_cat_names=True,
#         )

#         self.encoder.fit(X)

#         return self

#     def transform(self, X:pd.DataFrame):
#         X_out = X.copy()
#         X_out = self.encoder.transform(X_out)
#         return X_out


# Models
# ver de agregar otros modelos:
# catboost, lightgbm, XGboost, etc.
class SklearnModelTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, name:str, model):
        self.name = name
        self.model = model

    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None):
        self.model.fit(X, y)
        return self 

    def transform(self, X:pd.DataFrame):
        X = X.copy()

        y_pred = self.model.predict(X)

        X[f'{self.name}_prediction'] = y_pred

        return X

# Core
class DataCheckPoint(TransformerMixin, BaseEstimator):
    def __init__(self, save_path:str):
        self.save_path = save_path

    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None):
        return self 

    def transform(self, X:pd.DataFrame):
        X = X.copy()

        X.to_csv(self.save_path)

        return X

# Core
class DataTypeCheck(TransformerMixin, BaseEstimator):

    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None):

        self.data_types = X.dtypes.to_dict()
        return self 

    def transform(self, X:pd.DataFrame):

        X = X.copy()

        for col_name, typ in self.data_types.items():
            if col_name not in X.columns:
                raise ValueError(f"Column {col_name} not in fitted DataFrame")
            if (typ != X[col_name].dtype):
                raise ValueError(f"Column {col_name} was fitted with datatype {typ} instead of {X[col_name].dtype}.")

        return X


# Core
class PivotTable(TransformerMixin, BaseEstimator):
    
    def __init__(self, index:str, columns:List[str], values:List[str], aggfunc:str=None):
        self.index  = index
        self.columns = columns
        self.values = values
        self.aggfunc = aggfunc

    def _do_pivote(self, X:pd.DataFrame):
        if self.aggfunc == None:
            X = X.pivot(
                index=self.index, 
                columns=self.columns, 
                values=self.values
            )
        else:
            X = pd.pivot_table(
                X, 
                index=self.index, 
                columns=self.columns, 
                values=self.values, 
                aggfunc=self.aggfunc, 
            )
        return X


    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None):
        X = X.copy()
        X = self._do_pivote(X)
        self.pivot_columns = list(X.columns)
        return self 

    def transform(self, X:pd.DataFrame):

        X = X.copy()
        X = self._do_pivote(X)

        X = pd.DataFrame(
            data = X.values,
            columns=self.pivot_columns,
            index = X.index
        )

        return X

# DataFrame.join(other, on=None, how='left', lsuffix='', rsuffix='', sort=False)


############### Deprecated ###############
class JoinTables(TransformerMixin, BaseEstimator):

    # pipeline.fit(X, y, name__kwarg=value))

    def __init__(self, table:pd.DataFrame, on:str, how:str, lsuffix:str='', rsuffix:str=''):
        self.table = table
        self.on  = on
        self.how = how
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix

    def _do_join(self, X:pd.DataFrame)-> pd.DataFrame:
        if self.aggfunc == None:
            X = X.pivot(
                index=self.index, 
                columns=self.columns, 
                values=self.values
            )
        else:
            X = pd.pivot_table(
                X, 
                index=self.index, 
                columns=self.columns, 
                values=self.values, 
                aggfunc=self.aggfunc, 
            )
        return X


    def fit(self, X:pd.DataFrame, y:pd.DataFrame=None, X2:pd.DataFrame=None):
        X = X.copy()
        X = self._do_join(X, X2)
        self.pivot_columns = list(X.columns)
        return self 

    def transform(self, X:pd.DataFrame):

        X = X.copy()

        X = self._do_join(X)

        X = pd.DataFrame(
            data = X.values,
            columns=self.pivot_columns,
            index = X.index
        )

        return X


from sklearn.pipeline import Pipeline
from sklearn.utils import _print_elapsed_time

class CustomPipeline(Pipeline):

    def __init__(self, steps:list=None, memory=None, verbose=False):

        if steps:
            super().__init__(steps, memory, verbose)
        else:
            self.memory = memory
            self.verbose = verbose

    def add(self, step):
        self.steps.append(step)
        if len(self.steps) == 1:
            super().__init__(self.steps, self.memory, self.verbose)
    
    def remove(self, step_name):
        for step in self.steps:
            if step[0] == step_name:
                self.steps.remove(step)

    def summary(self):
        for idx,step in enumerate(self.steps):
            print(f"******Step {idx}*******")
            print(step[0])
            print(step[1])
            print()


    def fit_debug(self, X, y=None, **fit_params):
        self.verbose = True
        self.fit(X, y)
        self.verbose = False
        return self

    def transform_debug(self, X):
        self.verbose = True
        X = X.copy()
        for idx, _, transformer in self._iter():
            with _print_elapsed_time("Pipeline", message=self._log_message(idx)):
                X = transformer.transform(X)
        self.verbose = False
        return X

    def save(self):
        pass

  
class PandasFeatureJoin(FeatureUnion):
    def __init__(self, transformer_list:list, on:str, how:str, n_jobs=None, transformer_weights=None, verbose=False):
        super().__init__(transformer_list, n_jobs=n_jobs, transformer_weights=transformer_weights, verbose=verbose)
        self.on = on
        self.how = how

    
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans,
                X=X,
                y=y,
                weight=weight,
                **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.join_dataframes(Xs)
        return Xs

    def join_dataframes(self, Xs):
        X = Xs[0]
        for Xi in Xs[1:]:
            X = X.join(Xi, on=self.on, how=self.how)

        return 

    def transform(self, X:pd.DataFrame):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            # Raise Error
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            # Raise Error
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.join_dataframes(Xs)
        return Xs








