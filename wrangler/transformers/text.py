#Text Transformers
from wrangler.base import AbstractTransformer
from wrangler.base import FunctionWrapper
import pandas as pd

from typing import Any, Union, List, AnyStr, Dict, Callable

from sklearn.utils.validation import check_is_fitted


class TextConverter(AbstractTransformer):
    def __init__(self, columns:list):
        self.columns = columns

    def _fit(self, X:pd.DataFrame):
        return self
    
    def _transform(self, X:pd.DataFrame):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].astype(str)
        return X
    

class TextFillNa(AbstractTransformer):
    def __init__(self, columns:list, value='sin_dato'):
        self.columns = columns
        self.value = value

    def _fit(self, X:pd.DataFrame):
        return self
    
    def _transform(self, X:pd.DataFrame, **kwargs):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].fillna(self.value)
        return X    
    

class TextKeepTopNCategories(AbstractTransformer):
    def __init__(self, column:str, max_cats:int, left_value:str = 'otra'):
        self.column = column
        self.max_cats = max_cats
        self.left_value = left_value

    def _fit(self, X:pd.DataFrame):
        cats = X[self.column].value_counts().index
        
        if len(cats) > self.max_cats:
            self.categories_ = cats[:self.max_cats]
        else:
            self.categories_ = cats
        
        return self
    
    def _transform(self, X:pd.DataFrame):
        
        check_is_fitted(self, 'categories_')
        X = X.copy()
        X[self.column] = X[self.columns].apply(lambda x: x if x in self.categories_ else self.left_value)
        
        return X


class TextSplitByCharacter(AbstractTransformer):
    def __init__(self, column:str, split_string:str = " "):
        self.column = column
        self.split_string = split_string
    
    def _fit(self, X:pd.DataFrame):
        return self
    
    def _transform(self, X:pd.DataFrame):
        X=X.copy()
        X = X[self.column].str.split(self.split_string, expand=True)
        X.columns = [f'{self.column}_splited_{i}' for i in X.columns]
        
        return X


class ModeImputer(AbstractTransformer):
    
    def __init__(self, column:str):
        self.column = column       
    
    
    def _fit(self, X:pd.DataFrame):
        try:
            X = X.copy()
            self.impute_value = X[self.column].mode()[0]
        except Exception as e:
            print(e)
        finally:
            return self
            
            
    def _transform(self, X:pd.DataFrame):

        X = X.copy()
        X[self.column] = X[self.column].fillna(self.impute_value)
        
        return X

      
class GroupModeImputer(AbstractTransformer):
    
    def __init__(self, feature_name:str, group_cols:list):
        self.feature_name = feature_name   
        self.group_cols = group_cols
    
    
    def _fit(self, X:pd.DataFrame):

        X = X.copy()
        self.impute_map = X.groupby(self.group_cols)[self.feature_name].agg(
            lambda x:x.value_counts().index[0]
        ).reset_index(drop=False)

        return self
            
    def _transform(self, X:pd.DataFrame):
    
        X = X.copy()
        for _, row in self.impute_map.iterrows():
            ind = (X[self.group_cols] == row[self.group_cols]).all(axis=1)
            X.loc[ind, self.feature_name] = X.loc[ind, self.feature_name].fillna(row[self.feature_name])
        
        return X



class OneHotEncoderTransformer(AbstractTransformer):
    
    def __init__(self, column:str):
        self.column = column
        
    def _fit(self, X:pd.DataFrame):
        from sklearn.preprocessing import OneHotEncoder
        X = X.copy()
        self.encoder = OneHotEncoder(sparse=False)
        self.encoder.fit(X[[self.column]])
        self.one_hot_columns = self.encoder.get_feature_names([self.column])
        return self

    def _transform(self, X:pd.DataFrame):
        X = X.copy()
        X_encoded = X[[self.column]].copy()
        X_encoded = pd.DataFrame(
            data=self.encoder.transform(X[[self.column]]), 
            columns=self.one_hot_columns, 
            index=X.index)
        # X_encoded.columns = self.one_hot_columns
        X.drop(columns=[self.column], inplace=True)
        X_out = pd.concat([X, X_encoded], axis=1)
        return X_out


class OrdinalEncoderTransformer(AbstractTransformer):
    def __init__(self, columns:List[str]):
        self.columns = columns
        
    def _fit(self, X:pd.DataFrame):
        from sklearn.preprocessing import OrdinalEncoder
        X = X.copy()
        self.encoder = OrdinalEncoder()
        self.encoder.fit(X[self.columns])
        return self

    def _transform(self, X:pd.DataFrame):
        X = X.copy()
        X[self.columns] = self.encoder.transform(X[self.columns])
        return X
