#Date and Datetime Transformers
# Core Transformers for General Use
from wrangler.base import AbstractTransformer
from wrangler.base import FunctionWrapper
import pandas as pd

from typing import Any, Union, List, AnyStr, Dict, Callable


class TimestampToDate(AbstractTransformer):
    """Create date column from timestamp column"""

    def __init__(self, column:AnyStr='timestamp', new_column:AnyStr='date', unit:AnyStr='ns'):
        self.column = column
        self.new_column = new_column
        self.unit = unit

    def _fit(self, X:pd.DataFrame):
        # chequear que existe timestamp
        pass

    def _transform(self, X:pd.DataFrame):
        X = X.copy()
        X[self.new_column] = pd.to_datetime(X[self.column],
                                             unit=self.unit).dt.date.astype('datetime64[ns]')
        return X


class TimestampToDatetime(AbstractTransformer):
    """Create datetime column from timestamp column"""

    def __init__(self, column='timestamp', new_column='datetime', unit='ns'):
        self.column = column
        self.new_column = new_column
        self.unit = unit

    def _fit(self, X:pd.DataFrame):
        # chequear que existe timestamp
        pass

    def _transform(self, X:pd.DataFrame):
        X = X.copy()
        X[self.new_column] = pd.to_datetime(X[self.column],
                                             unit=self.unit).astype('datetime64[ns]')
        return X


class FillMissingDates(AbstractTransformer):
    """Fill missing dates in df by creating a new row for each missing date."""

    def __init__(self, start=None, end=None, column='date'):
        self.start = start
        self.end = end
        self.column = column

    def _fit(self, X:pd.DataFrame):
        # chequear que column sea un datetime?
        pass

    def _transform(self, X:pd.DataFrame):
        X = X.copy()
        
        start = self.start or X[self.column].min()
        end = self.end or X[self.column].max()
        idx = pd.date_range(start, end) # create complete date series from start to end
        X = X.sort_values(by=self.column)
        X = X.set_index(self.column)
        X = X.reindex(idx) # reindex by complete date series
        X = X.reset_index()
        X = X.rename(columns={'index': self.column})
        return X


class DayOfWeek(AbstractTransformer):
    """Compute day of week from dates."""

    def __init__(self, column:AnyStr='date', new_column:AnyStr='day_of_week'):
        self.column = column
        self.new_column = new_column

    def _fit(self, X:pd.DataFrame):
        # chequear que column sea un datetime?
        pass

    def _transform(self, X:pd.DataFrame):
        X = X.copy()
        X[self.new_column] = X[self.column].dt.dayofweek
        return X


class Month(AbstractTransformer):
    """Compute month of year from dates."""

    def __init__(self, column='date', new_column='month'):
        self.column = column
        self.new_column = new_column

    def _fit(self, X:pd.DataFrame):
        # chequear que column sea un datetime?
        pass

    def _transform(self, X:pd.DataFrame):
        X = X.copy()
        X[self.new_column] = X[self.column].dt.month
        return X

# No es explicitamente con dates pero se usa normalmente con dates.
# ver si queda aca.
class ShiftColumn(AbstractTransformer):
    """
    Create shifted features.

    It is mostly use with datetime indeces.
    """

    def __init__(self, columns, shifts:List[int]=[1]):
        self.columns = columns
        self.shifts = shifts

    def _fit(self, X:pd.DataFrame):
        pass   

    def _transform(self, X:pd.DataFrame):
        X = X.copy()
        for col in self.columns:
            for i in self.shifts:
                X['{:s}_shift{:d}'.format(col, i)] = X[col].shift(i)
        return X

# No es explicitamente con dates pero se usa normalmente con dates.
# ver si queda aca.
class ConsecutiveDifference(AbstractTransformer):
    """Compute difference of consecutive rows and shift.
    Examples:
        shift=1 is the difference between today and yesterday
        shift=2 is the difference between yesterday and the day before
    """

    def __init__(self, columns, shifts=[1]):
        self.columns = columns
        self.shifts = shifts

    def _fit(self, X:pd.DataFrame):
        pass  

    def _transform(self, X:pd.DataFrame):
        X = X.copy()
        for col in self.columns:
            diff = X[col] - X[col].shift() # consecutive diff column
            for i in self.shifts:
                X['{:s}_condiff{:d}'.format(col, i)] = diff.shift(i-1) # add shifted diff columns
        return X


class CumulativeDifference(AbstractTransformer):
    """Compute cumulative difference of shifted columns.
    Examples:
        shift=1 is the difference between today and yesterday
        shift=2 is the difference between today and 2 days ago
    """

    def __init__(self, columns, shifts=[1]):
        self.columns = columns
        self.shifts = shifts

    def _fit(self, X:pd.DataFrame):
        pass  

    def _transform(self, X:pd.DataFrame):
        X = X.copy()    
        for col in self.columns:
            for i in self.shifts:
                X['{:s}_cumdiff{:d}'.format(col, i)] = X[col] - X[col].shift(i)
        return X


class RollingMeanDifference(AbstractTransformer):
    """Compute difference from moving average features.
    Examples:
      shift=1, window=2 is the difference between today and the mean of yesterday and 2 days ago
      shift=2, window=2 is the difference between yesterday and the mean of 2 days and 3 days ago
    """

    def __init__(self, columns, shifts=[1], window=2):
        self.columns = columns
        self.shifts = shifts
        self.window = window

    def _fit(self, X:pd.DataFrame):
        pass  

    def transform(self, X:pd.DataFrame):
        X = X.copy()
        for col in self.columns:
            diff = X[col] - X[col].rolling(window=self.window).mean().shift() # compute diff from moving avg
            for i in self.shifts:
                X['{:s}_rmdiff{:d}'.format(col, i)] = diff.shift(i-1)
        return X