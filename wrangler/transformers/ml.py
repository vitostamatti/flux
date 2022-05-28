from sklearn import model_selection
from wrangler.base import AbstractDataset, AbstractTransformer
from wrangler.base import FunctionWrapper
import pandas as pd
import numpy as np
import os
import dill

from sklearn.model_selection import train_test_split

from typing import Any, Union, List, AnyStr, Dict, Callable


class ModelTransformer(AbstractTransformer):
    def __init__(self, name:str, model, fit_params=None,
                 filename:str= None):
        self.name = name
        self.model = model
        self.fit_params = fit_params
        self.filename = filename

    def _fit(self, X, y):
        if self.fit_params is not None:
            self.model.fit(X, y, **self.fit_params)
        else:
            self.model.fit(X, y)
        return self 

    def _transform(self, X, y=None):
        y_pred = self.model.predict(X)
        return y_pred



class SklearnModelTransformer(AbstractTransformer):
    def __init__(self, name:str, model,
                 filename:str= None
                 ):
        self.name = name
        self.model = model
        self.filename = filename

    def _fit(self, X:pd.DataFrame, y:pd.DataFrame=None):
        self.model.fit(X, np.ravel(y))

        if self.filename:
          dill.dump(self.model, open(f'{self.filename}.plk', 'wb'))
        return self

    def _transform(self, X:pd.DataFrame, y:pd.DataFrame=None):
        X = X.copy()
        y_pred = self.model.predict(X)
        # Retornar solo el y_pred
        # X[f'{self.name}_prediction'] = y_pred
        return pd.DataFrame(y_pred, columns=[f'{self.name}_prediction'])


class SplitFeaturesTarget(AbstractTransformer):

    def __init__(self, target:str):
        self.target = target

    def _fit(self, X:pd.DataFrame):
        return self

    def _transform(self, X:pd.DataFrame):
        X_in = X.copy()
        if self.target in X_in.columns:
            features = X_in.columns.to_list()
            features.remove(self.target)
            X_out = X_in[features]
            y_out = X_in[[self.target]]
        else:
            X_out = X_in.copy()
            X_in[self.target]=np.NaN
            y_out = X_in[self.target]
        return X_out, y_out


class SplitTrainTest(AbstractTransformer):
    # Candidato a ser volado! 
    def __init__(self, test_size:float):
        self.test_size = test_size

    def _fit(self, X:pd.DataFrame, y:pd.DataFrame):
        return self

    def _transform(self, X:pd.DataFrame, y:pd.DataFrame):
        X = X.copy()
        y = y.copy()
        X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=self.test_size)
        return X_train, y_train, X_test, y_test



from sklearn.model_selection import cross_validate



class RegressionModelEvaluator(AbstractTransformer):
    
    def __init__(self, model, name=None,) -> None:
        self.name = type(self).__name__ if name is None else name
        self.model = model
        self.metrics = ['r2','neg_mean_squared_error','neg_root_mean_squared_error']

    def _fit(self, X:pd.DataFrame, y:pd.DataFrame):
        X = X.copy()
        y = y.copy()
        scores = cross_validate(self.model, X, np.ravel(y), scoring=self.metrics, return_train_score=True) 
        self.results = pd.DataFrame(scores)
        return self 

    def _transform(self, X:pd.DataFrame, y:pd.DataFrame=None):
        X = X.copy()
        y = y.copy()
        return self.results



class ClassificationModelEvaluator(AbstractTransformer):

    def __init__(self, model, name=None,) -> None:
        self.name = type(self).__name__ if name is None else name
        self.model = model
        self.metrics = ['accuracy','precision', 'recall','f1','roc_auc']

    def _fit(self, X:pd.DataFrame, y:pd.DataFrame=None):
        X = X.copy()
        y = y.copy()
        scores = cross_validate(self.model, X, np.ravel(y), scoring=self.metrics, return_train_score=True) 
        self.results = pd.DataFrame(scores)
        return self

    def _transform(self, X:pd.DataFrame, y:pd.DataFrame=None):
        X = X.copy()
        y = y.copy()
        return self.results

# imports
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

class SelectBestKMeansInertiaModel(AbstractTransformer):

    def __init__(self, max_k, parallel=False):
        self.max_k = max_k
        self.parallel = parallel

    def compute_interia(self, data, k, alpha_k=0.02):
        inertia_o = np.square((data - data.mean(axis=0))).sum()
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
        return scaled_inertia

    def select_best_k_parallel(self, data, max_k):
        ans = Parallel(n_jobs=-1,verbose=10)(
            delayed(self.compute_interia)(data, k) for k in range(2,max_k+1)
            )
        ans = list(zip(range(2,max_k+1),ans))
        results = pd.DataFrame(ans, columns = ['k','Scaled Inertia']).set_index('k')
        best_k = results.idxmin()[0]
        return best_k, results

    def select_best_k(self, data, max_k):
        ans = []
        for k in range(2, max_k+1):
            scaled_inertia = self.compute_interia(data, k)
            ans.append((k, scaled_inertia))
        results = pd.DataFrame(ans, columns = ['k','Scaled Inertia']).set_index('k')
        best_k = results.idxmin()[0]
        return best_k, results

    def _fit(self, X):
        if self.parallel:
            self.best_k, self.results = self.select_best_k_parallel(X, self.max_k)
        else:
            self.best_k, self.results = self.select_best_k(X, self.max_k)

        return self

    def _transform(self, X):
        return self.best_k, self.results


class KMeansModelTransformer(AbstractTransformer):

    def __init__(self, n_clusters=3, n_init=300, max_iter=1000):
        self.model = KMeans(
            n_clusters = n_clusters,
            random_state = 100,
            n_init=n_init,
            max_iter=max_iter
            )

    def _fit(self, X:pd.DataFrame):
        self.model.fit(X)
        self.labels = self.model.labels_
        return self

    def _transform(self, X:pd.DataFrame):
        X = X.copy()
        cluster = self.model.predict(X)
        return pd.DataFrame(cluster, columns=['cluster'], index=X.index)



from dataclasses import dataclass
from scipy.stats import chi2_contingency, ks_2samp
from typing import Callable, Dict, List, Tuple, Union

@dataclass
class DataDriftResult:
    is_drift: bool = False
    distance: float= None
    p_val: float = .0
    threshold: float = .0


class DataDriftTransformer(AbstractTransformer):

    def __init__(self, columns:list = None, p_val:float = .05):
        self.columns = columns
        self.p_val = p_val


    def _get_categories_per_feature(self, X):
        category_map = []
        for i, row in enumerate(X.dtypes):
            if row == 'object':
                category_map.append(i)
        categories_per_feature = {f: None for f in list(category_map)}
        return categories_per_feature


    def _compute_stats(self, X):
        num_stats = X.describe().T
        cat_stats = pd.DataFrame(X.iloc[:,self.cat_vars].nunique(), columns=['nunique'])
        full_stats = pd.concat([num_stats, cat_stats]).drop(columns=['count'])

        return full_stats


    def _fit(self, X_ref:pd.DataFrame, X_comp:pd.DataFrame):

        if self.columns:
            X_ref = X_ref[self.columns].copy()

        x_ref = X_ref.values

        self.feature_names = X_ref.columns
        self.n_features = len(X_ref.columns)

        categories_per_feature = self._get_categories_per_feature(X_ref)

        self.x_ref_categories, self.cat_vars = {}, []

        vals = list(categories_per_feature.values())
        int_types = (int, np.int16, np.int32, np.int64)
        if all(v is None for v in vals):
            x_flat = x_ref.reshape(x_ref.shape[0], -1)
            categories_per_feature = {f: list(np.unique(x_flat[:, f]))
                                        for f in categories_per_feature.keys()}

        self.x_ref_categories = categories_per_feature
        self.cat_vars = list(self.x_ref_categories.keys())
        self.ref_stats =  self._compute_stats(X_ref)

        return self


    def _transform(self, X_ref:pd.DataFrame, X_comp:pd.DataFrame):

        if self.columns:
            X_ref = X_ref[self.columns].copy()
            X_comp = X_comp[self.columns].copy()

        preds = self._predict(X_ref.values, X_comp.values)
        X_out = []
        for f in range(self.n_features):
            stat = 'Chi2' if f in list(self.x_ref_categories.keys()) else 'K-S'
            fname = self.feature_names[f]
            is_drift = preds.is_drift[f]
            stat_val, p_val = preds.distance[f], preds.p_val[f]
            X_out.append([fname,is_drift,stat,stat_val,p_val])

        X_out = pd.DataFrame(X_out, columns=['feature','is_drift','test_type','test_val','p_val']).set_index('feature')
        out_ref_stats = self.ref_stats
        out_ref_stats.columns = [f"ref_{col}" for col in out_ref_stats.columns]
        out_comp_stats = self._compute_stats(X_comp)
        out_comp_stats.columns = [f"comp_{col}" for col in out_comp_stats.columns]
        X_out = X_out.join(out_ref_stats).join(out_comp_stats)

        return X_out


    def feature_score(self, x_ref: np.ndarray, x_comp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute K-S or Chi-Squared test statistics and p-values per feature.
        Parameters
        ----------
        x_ref
            Reference instances to compare distribution with.
        x_comp
            Batch of instances.
        Returns
        -------
        Feature level p-values and K-S or Chi-Squared statistics.
        """
        x_ref = x_ref.reshape(x_ref.shape[0], -1)
        x_comp = x_comp.reshape(x_comp.shape[0], -1)

        # apply counts on union of categories per variable in both the reference and test data
        if self.cat_vars:
            x_categories = {f: list(np.unique(x_ref[:, f])) for f in self.cat_vars}
            all_categories = {f: list(set().union(self.x_ref_categories[f], x_categories[f]))  # type: ignore
                              for f in self.cat_vars}
            x_ref_count = self._get_counts(x_ref, all_categories)
            x_count = self._get_counts(x_comp, all_categories)

        p_val = np.zeros(self.n_features, dtype=np.float32)
        dist = np.zeros_like(p_val)
        for f in range(self.n_features):
            if f in self.cat_vars:
                contingency_table = np.vstack((x_ref_count[f], x_count[f]))
                dist[f], p_val[f], _, _ = chi2_contingency(contingency_table)
            else:
                dist[f], p_val[f] = ks_2samp(x_ref[:, f], x_comp[:, f], alternative='two-sided', mode='asymp')
        return p_val, dist
            # correction: str = 'bonferroni',

    def _get_counts(self, x: np.ndarray, categories: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """
        Utility method for getting the counts of categories for each categorical variable.
        """
        return {f: [(x[:, f] == v).sum() for v in vals] for f, vals in categories.items()}


    def _predict(self, x_ref:np.ndarray, x_comp:np.ndarray):
        """
        Predict whether a batch of data has drifted from the reference data.
        """
        # compute drift scores
        p_vals, dist = self.feature_score(x_ref, x_comp)

        # values below p-value threshold are drift
        drift_pred = (p_vals < self.p_val).astype(int)

        # populate drift result
        ddr = DataDriftResult()
        ddr.is_drift = drift_pred
        ddr.p_val = p_vals
        ddr.threshold = self.p_val
        ddr.distance = dist
        return ddr 


### Selectores de Variables
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold

# !pip3 install git+https://github.com/smazzanti/mrmr
# from mrmr import mrmr_regression
# from boruta import BorutaPy
# from mlxtend.feature_selection import ExhaustiveFeatureSelector



class ConstantFeatureSelector(AbstractTransformer):
    
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def _fit(self, X:pd.DataFrame, y:pd.DataFrame=None):
        X = X.copy()
        # Get number of rows in X
        num_rows = X.shape[0]

        # Get column labels
        features = X.columns.tolist()

        # Make a dict to store the fraction describing the value that occurs the most
        constant_per_feature = {label: X[label].value_counts().iloc[0]/num_rows for label in features}

        # Determine the features that contain a fraction of missing values greater than the specified threshold
        self.selected_features = [
            feature for feature in features if constant_per_feature[feature] < self.threshold
        ]
    
        return self
    
    def _transform(self, X:pd.DataFrame):
        return X[self.selected_features]

    
    
class CorrelationFeatureSelector(AbstractTransformer):

    def __init__(self, threshold=0.9):
        self.threshold = threshold
        
    def _fit(self, X, y):
        X = X.copy()
        y = y.copy()
        # Make correlation matrix

        X_y = pd.concat([X,y],axis=1)
        
        corr_matrix = X_y.corr(method = "spearman").abs()

        corr_df = pd.DataFrame(corr_matrix, columns=X_y.columns, index=X_y.columns)
        corr_df=corr_df.iloc[:-1,-1:]

        self.selected_features = list(corr_df[corr_df.iloc[:,-1]>=self.threshold].index)

    
        print(f"Features selected by Correlation: {self.selected_features} \n")  

        return self
    
    def _transform(self, X, y=None):
        return X[self.selected_features]

    
  
class RFEFeatureSelector(AbstractTransformer):
    
    def __init__(self, n_features, estimator = LinearRegression()):
        
        self.n_features = n_features
        self.estimator = estimator 

    def _fit(self, X, y=None):
        
        feature_selector = RFECV(
            self.estimator, step=1, cv=3, verbose=0, 
            min_features_to_select=self.n_features
        )
        feature_selector.fit(X, np.ravel(y))
        features = X.columns
        
        self.selected_features = features[feature_selector.support_].tolist()
        print(f"Features selected by RFE: {self.selected_features} \n")  
        
        return self
    
    def _transform(self, X, y=None):
        
        return X[self.selected_features]

    
    
class KBestFeatureSelectorRegression(AbstractTransformer):
    
    def __init__(self, n_features):
        self.n_features = n_features
    

    def _fit(self, X, y):
        
        X = X.copy()
        feature_names = X.columns
        features_selected = []
        
        kf = KFold(n_splits = 3)
        for train_index, val_index in kf.split(X):
            X_train_fold , X_val_fold = X.iloc[train_index,:], X.iloc[val_index,:]
            y_train_fold , y_val_fold = y.iloc[train_index,:], y.iloc[val_index,:]
            
            f_selector = SelectKBest(mutual_info_regression, k=self.n_features)
            f_selector.fit(X_train_fold, np.ravel(y_train_fold))
            features_selected_fold = feature_names[f_selector.get_support()]
            features_selected.append(list(features_selected_fold))
            
        self.selected_features = set([feat for feat_sel in features_selected for feat in feat_sel])
        
        print(f"Features selected by Select KBest: {self.selected_features} \n")   
        
        return self

    def _transform(self, X, y=None):
        
        return X[self.selected_features]
    

class KBestFeatureSelectorClassification(AbstractTransformer):
    
    def __init__(self, n_features):
        self.n_features = n_features
    

    def _fit(self, X, y):
        
        X = X.copy()
        feature_names = X.columns
        features_selected = []
        
        kf = KFold(n_splits = 3)
        for train_index, val_index in kf.split(X):
            X_train_fold , X_val_fold = X.iloc[train_index,:], X.iloc[val_index,:]
            y_train_fold , y_val_fold = y.iloc[train_index,:], y.iloc[val_index,:]
            
            f_selector = SelectKBest(mutual_info_classif, k=self.n_features)
            f_selector.fit(X_train_fold, np.ravel(y_train_fold))
            features_selected_fold = feature_names[f_selector.get_support()]
            features_selected.append(list(features_selected_fold))
            
        self.selected_features = set([feat for feat_sel in features_selected for feat in feat_sel])
        
        print(f"Features selected by Select KBest: {self.selected_features} \n")   
        
        return self

    def _transform(self, X, y=None):
        
        return X[self.selected_features]

    
class BackwardForwardFeatureSelector(AbstractTransformer):
  
    def __init__(self, n_features, estimator=LinearRegression()):
        
        self.n_features = n_features
        self.estimator = estimator
        
    def _fit(self, X, y):
        
        X = X.copy()
        selected_features = []
        feature_names = X.columns
        
        # Cross Validate
        kf = KFold(n_splits = 3)
        for train_index, val_index in kf.split(X):
            X_train_fold , X_val_fold = X.iloc[train_index,:], X.iloc[val_index,:]
            y_train_fold , y_val_fold = y.iloc[train_index,:], y.iloc[val_index,:]      

            self.estimator.fit(X_train_fold, np.ravel(y_train_fold))

            sfs_forward = SequentialFeatureSelector(
                self.estimator, 
                n_features_to_select=self.n_features, 
                direction="forward"
            ).fit(X_train_fold, np.ravel(y_train_fold))

            sfs_backward = SequentialFeatureSelector(
                self.estimator, 
                n_features_to_select=self.n_features, 
                direction="backward"
            ).fit(X_train_fold, np.ravel(y_train_fold))

            selected_features_fold = set.union(
                set(feature_names[sfs_forward.get_support()]),
                set(feature_names[sfs_backward.get_support()])
            )
            
            selected_features.append(list(selected_features_fold))
            
        self.selected_features = set([feat for feat_sel in selected_features for feat in feat_sel])

        print(
            f"Features selected by union of back and forward sequential selection: {self.selected_features} \n"
        )   
        
        return self
    
    def _transform(self, X, y=None):

        return X[self.selected_features]


class VIFFeatureSelector(AbstractTransformer): 
    
    def __init__(self, threshold=5.0):
        
        self.threshold = threshold
        
    def _fit(self, X, y=None):
        
        variables = list(range(X.shape[1]))
        dropped = True
        while dropped:
            dropped = False
            vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
                   for ix in range(X.iloc[:, variables].shape[1])]

            maxloc = vif.index(max(vif))
            if (max(vif) > self.threshold) & (len(variables) > 1):
                # print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                #       '\' at index: ' + str(maxloc))
                del variables[maxloc]
                dropped = True

        self.selected_features = list(X.columns[variables])
        print(f"Features selected by VIF: {self.selected_features} \n")  
        return self
    
    def _transform(self, X, y=None):
        return X[self.selected_features]

    

class FromModelFeatureSelector(AbstractTransformer):
    
    def __init__(self, n_features, estimator=ElasticNet()):
        
        self.n_features = n_features
        self.estimator = estimator

    def _fit(self, X, y):
        
        selected_features = []

        # Cross Validate
        kf = KFold(n_splits = 3)
        for train_index, val_index in kf.split(X):
            X_train_fold , X_val_fold = X.iloc[train_index,:], X.iloc[val_index,:]
            y_train_fold , y_val_fold = y.iloc[train_index,:], y.iloc[val_index,:] 
            

            feature_selector = SelectFromModel(
                self.estimator, 
                max_features = self.n_features
            )
            feature_selector.fit(X_train_fold, np.ravel(y_train_fold))
            
            selected_features_fold = X.loc[:,feature_selector.get_support()].columns
            selected_features.append(selected_features_fold)
            
        self.selected_features = set([feat for feat_sel in selected_features for feat in feat_sel])

        print(
           f'Features selected from Model {self.estimator}: {self.selected_features} \n'
        )   
           
        return self
    
    def _transform(self, X, y=None):
        
        return X[self.selected_features]    


# class MRMRFeatureSelector(AbstractTransformer):

#     def __init__(self, n_features):
#         self.n_features = n_features
        
#     def fit(self, X, y):
#         self.selected_features = mrmr_regression(X, y, K = self.n_features)
#         print(f"Features selected by MRMR: {self.selected_features} \n")  

#     def transform(self, X):
#         return X[self.selected_features]

    
    
# class ExhaustiveSearchFeatureSelector(AbstractTransformer):
    
#     def __init__(self, n_features, estimator=LinearRegression()):
#         self.n_features = n_features
#         self.estimator = estimator

        
#     def fit(self, X, y):
#         feature_selector = ExhaustiveFeatureSelector(
#             self.estimator, 
#             min_features=self.n_features, 
#             max_features=self.n_features, 
#             print_progress=True, 
#             scoring='r2', 
#             cv=3, 
#             n_jobs=-1, 
#         )
#         feature_selector.fit(X, y)
#         features = X.columns
#         self.selected_features = list(feature_selector.best_feature_names_)  
#         return self
    
    
#     def transform(self, X):
#         return X[self.selected_features]

    

# class BorutaFeatureSelector(AbstractTransformer):
    
#     def __init__(self, alpha = 0.05):
#         self.alpha = alpha
    
#     def fit(self, X, y=None):
#         X = X.copy()
        
#         selected_features = []
        
#         # Cross Validate
#         rf = RandomForestRegressor(n_jobs=-1, n_estimators=200, max_depth=3)
#         feat_selector = BorutaPy(
#             rf, n_estimators='auto', alpha=self.alpha,
#             verbose=0, random_state=42)
        
#         kf = KFold(n_splits = 3)
#         for train_index, val_index in kf.split(X):
            
#             X_train_fold , X_val_fold = X.iloc[train_index,:], X.iloc[val_index,:]
            
#             y_train_fold , y_val_fold = y[train_index], y[val_index]      
            
#             feat_selector.fit(X_train_fold.values, y_train_fold)
            
#             selected_features_fold = X_train_fold.columns[feat_selector.support_].to_list()
            
#             selected_features.append(selected_features_fold)
    
    
#         self.selected_features = set([
#             feat for feat_sel in selected_features for feat in feat_sel
#         ])
        
#         print(f"Features selected by Boruta: {self.selected_features} \n")  
        
#         return self
    
#     def transform(self, X):
#         return X[self.selected_features]
