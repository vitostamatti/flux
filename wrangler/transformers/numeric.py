#Numeric Transformers
from wrangler.base import AbstractTransformer
from wrangler.base import FunctionWrapper
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest


from typing import Any, Union, List, AnyStr, Dict, Callable


class ClipTransformer(AbstractTransformer):
    """[summary]

    Attributes:
        column (str): [description]
        min_value (float): [description]
        max_value (float): [description]
    """ 

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
    
    def _fit(self, X:pd.DataFrame):
        return self
        
    def _transform(self, X:pd.DataFrame):
        X = X.copy()
        X[self.column] = X[self.column].apply(lambda x: self._clip_func(x))
        return X


class OutliersQuantileTransformer(AbstractTransformer):
    """[summary]

    Attributes:
        column (str): [description]
        q_low (float): [description]
        q_up (float): [description]
        drop (bool, optional): [description]. Defaults to False.
    """  
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
    
    def _fit(self, X:pd.DataFrame, y:pd.DataFrame=None):
        X = X.copy()
        self.q_low_value = X[self.column].quantile(self.q_low)
        self.q_up_value = X[self.column].quantile(self.q_up)
        return self

    def _transform(self, X:pd.DataFrame):
        X = X.copy()
        X[self.column] = X[self.column].apply(lambda x: self._clip_func(x))
        return X


class GroupImputer(AbstractTransformer):
    """Class used for imputing missing values in a 
    pd.DataFrame using either mean or median of a group.

    Attributes:
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
    
    def _fit(self, X:pd.DataFrame):
        impute_map = X.groupby(self.group_cols)[self.target].agg(
            self.metric
        ).reset_index(drop=False)
        self.impute_map = impute_map
        return self 
    
    def _transform(self, X:pd.DataFrame):     
        X = X.copy()
        
        for index, row in self.impute_map.iterrows():
            ind = (X[self.group_cols] == row[self.group_cols]).all(axis=1)
            X.loc[ind, self.target] = X.loc[ind, self.target].fillna(row[self.target])
        
        return X


class Normalizer(AbstractTransformer):
    pass


class LogTransform(AbstractTransformer):
    pass

class ScalerTransformer(AbstractTransformer):
    def __init__(self, columns:list, mode ='standard') -> None:
        self.columns = columns
        self.mode = mode

    def _fit(self,X:pd.DataFrame):
        X = X.copy()

        if self.mode=='standard':
            self.scaler = StandardScaler()
        elif self.mode=='robust':
            self.scaler = RobustScaler()
        elif self.mode=='minmax':
            self.scaler = MinMaxScaler()
        else:
            raise Exception(f"Mode {self.mode} not supported")
        
        self.scaler.fit(X[self.columns])

        return self

    def _transform(self, X:pd.DataFrame):
        X = X.copy()

        X[self.columns] = self.scaler.transform(X[self.columns])

        return X


class IsolationForestTransformer(AbstractTransformer):
    """[summary]

    Attributes:
    
    """  
    def __init__(self,columns:str, n_estimators=100, 
                max_features=1.0, contamination='auto', 
                max_samples='auto'):
      
        self.columns = columns
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.contamination = contamination
        self.max_samples = max_samples
  
    def _fit(self, X:pd.DataFrame, y:pd.DataFrame=None):
        X = X.copy()
        self.ift = IsolationForest(
            random_state=1, 
            n_estimators=self.n_estimators, 
            max_features=self.max_features,
            contamination=self.contamination,
            max_samples=self.max_samples
        )
        self.ift.fit(X[self.columns])

        return self

    def _transform(self, X:pd.DataFrame):
        X = X.copy()
        mask = self.ift.predict(X[self.columns]) == 1
        outliers = X[~mask]
        inliers = X[mask]
        self._logger.info(f'{outliers.shape[0]} instances where marked as outliers out of {X.shape[0]}')
        return inliers, outliers


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class DimensionReducerTransformer(AbstractTransformer):
    """[summary]

    Attributes:
    
    """  
    def __init__(self, n_components, mode='pca'):
        self.n_components = n_components
        self.mode = mode

    def _fit(self, X:pd.DataFrame):
        X = X.copy()

        if self.mode == "pca":
            self.reducer = PCA(n_components=self.n_components, random_state=1)
        elif self.mode == 'tsne':
            self.reducer = TSNE(n_components=self.n_components, random_state=1)
        else:
            raise Exception(f"Mode {self.mode} not supported")

        # self.reducer.fit(X)

        return self

    def _transform(self, X:pd.DataFrame):
        X = X.copy()
        X_reduced = self.reducer.fit_transform(X)
        X_reduced = pd.DataFrame(X_reduced, columns=[f'{self.mode}_{i+1}' for i in range(X_reduced.shape[1])], index=X.index)
        return X_reduced
 

from scipy.stats import skew

class SkewedDataEvaluator(AbstractTransformer):

    '''
    Check numerical features distribution skewness.
    Apply transformation if needed.

    Hint:
        For normally distributed data, the skewness should be about zero. 
        For unimodal continuous distributions, a skewness value greater than zero means 
        that there is more weight in the right tail of the distribution.

    Attributes:
        threshold (float): amount of tolerable skewness
        function (Callable, optional): the transformation
            to apply in skewed variables. Defaults to np.log.
    '''

    def __init__(self, threshold:float, function:Callable=np.log) -> None:
        """[summary]

        Args:
            threshold (float): [description]
            function (Callable, optional): [description]. Defaults to np.log.
        """        
        self.threshold = threshold
        self.function = function
        self.skewed_cols = []

    def _fit(self, X:pd.DataFrame):
        X = X.copy()
        num_cols = X.select_dtypes('number').columns
        for col in num_cols:
            skewness = skew(X[col])
            if skewness > self.threshold or skewness < - self.threshold:
                self.skewed_cols.append(col)
        return self

    def _transform(self, X:pd.DataFrame):
        X = X.copy()
        for col in self.skewed_cols:
            X[col] = self.function(X[col])
            
        return X