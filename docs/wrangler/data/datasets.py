# Implementations of Dataset
import logging
from typing import Any, AnyStr
import io
import os
from abc import ABC
from abc import abstractmethod
import copy
import json
import pandas as pd
import pickle
import yaml



class AbstractDataset(ABC):
    """Abstract class to encapsulate all datasets load and save methods.
    """

    def __init__(self, decription:AnyStr=None) -> None:
        self.description = decription
   
    @property
    def _logger(self):
        return logging.getLogger(__name__)


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
        try:
            return self._load(*args,**kwargs)
        except Exception as e:
            self._logger.error(e)


    def save(self, *args, **kwargs):
        self._logger.info(f"Saving {str(self)}")
        try:
            return self._save(*args,**kwargs)
        except Exception as e:
            self._logger.error(e)

    def add_description(self, description):
        if description in self.description:
            self._logger.info(f"Replacing description of dataset {type(self).__name__}")    
        self.description = description
    

    @abstractmethod
    def _load(self, *args, **kwargs):
        pass

    @abstractmethod
    def _save(self, *args, **kwargs):
        pass



class MemoryDataset(AbstractDataset):
    """Allocates data in ram memory. Is practical for fast
    experiments, but does not allow data saving for later use.

    Attributes:
        data (pd.DataFrame, optional): [description]. Defaults to None.

    Example:
        .. code-block:: python

            cars_data = pd.DataFrame({
                "cars":["car1", "car2", "car3", "car4"],
            })
            cars = MemoryDataset(data=cars_data)

    """

    def __init__(self, data:Any = None,
                description:str = None ) -> None:

        super().__init__(description)
        self._data = data

    @property
    def data(self):
        return self._data

    def _load(self) -> Any:
        if self.data is None:
            raise Exception("This dataset does not have any data.")

        if isinstance(self.data, (pd.DataFrame, pd.Series)):
            data = self.data.copy()
        else:
            data = copy.deepcopy(self.data)
        return data

    def _save(self, data: Any):
        self._data = data


class CSVDataset(AbstractDataset):
    """Loads and Saves a .csv file from and to the local filesystem
    with specific parameters usign Pandas API.

    Attributes:
        filename (str): name of the file to load and save
        load_params (dict, optional): Pandas API load params. Defaults to None.
        save_params (dict, optional): Pandas API save params. Defaults to None.

    Example:
        .. code-block:: python

            cars = PandasDataset(filename='path/to/file.csv')

    """
    def __init__(self, filename:str,
                description:str=None ,load_params:dict=None,
                save_params:dict=None) -> None:

        super().__init__(description)
        self.filename = filename
        self.load_params = load_params if load_params else {}
        self.save_params = save_params if save_params else {}

    def _load(self):
        return pd.read_csv(self.filename, **self.load_params)


    def _save(self, data):
        if isinstance(data, pd.DataFrame):
            data.to_csv(self.filename, **self.save_params)
        else:
            raise ValueError("Data must be a `pd.DataFrame` object")



class ParquetDataset(AbstractDataset):
    """Loads and Saves a .parquet file from and to the local filesystem
    with specific parameters usign Pandas API.

    Attributes:
        filename (str): name of the file to load and save
        load_params (dict, optional): Pandas API load params. Defaults to None.
        save_params (dict, optional): Pandas API save params. Defaults to None.

    Example:
        .. code-block:: python

            cars = ParquetDataset(filename='path/to/file.parquet')

    """

    def __init__(self, filename: str,
                 description: str = None, load_params: dict = None,
                 save_params: dict = None) -> None:

        super().__init__(description)
        self.filename = filename
        self.load_params = load_params if load_params else {}
        self.save_params = save_params if save_params else {}

    def _load(self):
        return pd.read_parquet(self.filename, **self.load_params)


    def _save(self, data:pd.DataFrame):
        if isinstance(data, pd.DataFrame):
            data.to_parquet(self.filename, **self.save_params)
        else:
            raise ValueError("Data must be a `pd.DataFrame` object")


class ExcelDataset(AbstractDataset):
    """Loads and Saves an Excel file from and to the local filesystem
    with specific parameters usign Pandas API.

    Attributes:
        filename (str): name of the file to load and save
        load_params (dict, optional): Pandas API load params. Defaults to None.
        save_params (dict, optional): Pandas API save params. Defaults to None.

    Example:
        .. code-block:: python

            cars = PandasDataset(filename='path/to/file.xlsx')

    """
    def __init__(self, filename:str,
                load_params:dict=None, save_params:dict=None,
                description:str=None) -> None:

        super().__init__(description)
        self.load_params = load_params if load_params else {}
        self.save_params = save_params if save_params else {}
        self.filename = filename

    def _load(self):
        return pd.read_excel(self.filename, **self.load_params)
 
    def _save(self, data):
        if isinstance(data, pd.DataFrame):
            data.to_excel(self.filename, **self.save_params)
        else:
            raise ValueError("Data must be a `pd.DataFrame` object")



class JSONDataset(AbstractDataset):
    """Loads and Saves an json file from and to the local filesystem.

    Attributes:
        filename (str): name of the file to load and save.

    Example:
        .. code-block:: python

            cars = JSONDataset(filename='path/to/file.json')

    """
    def __init__(self, filename):
        self.filename = filename

    def _load(self):
        with open(self.filename, 'r') as f:
            data = json.load(f)
        return data

    def _save(self, data):
        with open(self.filename, 'w') as f:
            json.dump(data,f)


class PickleDataset(AbstractDataset):
    """Loads and Saves data file from and to the local filesystem using
    pickle library.

    Attributes:
        filename (str): name of the file to load and save.

    Example:
        .. code-block:: python
        
            cars = PickleDataset(filename='path/to/file.pkl')

    """
    def __init__(self, filename):
        self.filename = filename

    def _load(self):
        with open(self.filename, 'rb') as f:
            data = pickle.load(f)
        return data

    def _save(self, data):
        with open(self.filename, 'wb') as f:
            pickle.dump(data,f)


class YAMLDataset(AbstractDataset):
    """Loads and Saves YAML file from and to the local filesystem using
    pyyaml library.

    Attributes:
        filename (str): name of the file to load and save.

    Example:
        .. code-block:: python
        
            cars = YAMLDataset(filename='path/to/file.yml')

    """
    def __init__(self, filename):
        self.filename = filename

    def _load(self):
        with open(self.filename, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return data

    def _save(self, data):
        with open(self.filename, 'w') as f:
            yaml.dump(data,f)


    
class _BaseS3Dataset(AbstractDataset):
    def __init__(self, bucket:str, filename:str,
                access_key, secret_key, region_name:str='us-east-2', description:str=None) -> None:

        super().__init__(description)
        self.bucket = bucket
        self.filename = filename

        self.access_key = access_key
        self.secret_key = secret_key
        self.region_name = region_name

    def _get_s3_client(self):
        import boto3

        s3_client = boto3.client(
            service_name='s3',
            region_name=self.region_name,
            aws_access_key_id=os.environ.get(self.access_key),
            aws_secret_access_key=os.environ.get(self.secret_key),
        )
        return s3_client

    def _get_s3_object(self):
        s3_client = self._get_s3_client()
        
        response = s3_client.get_object(Bucket=self.bucket, Key=self.filename)

        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status == 200:
            self._logger.info(f"Successful S3 get_object response. Status - {status}")
        else:
            raise Exception(f"Unsuccessful S3 get_object response. Status - {status}")

        return response

    def _put_s3_object(self, out_buffer):
        s3_client = self._get_s3_client()
        response = s3_client.put_object(
                Bucket=self.bucket, Key=self.filename, Body=out_buffer.getvalue()
            )

        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status == 200:
            self._logger.info(f"Successful S3 put_object response. Status - {status}")
        else:
            raise Exception(f"Unsuccessful S3 put_object response. Status - {status}")




class S3CSVDataset(_BaseS3Dataset):
    """Loads and Saves a csv file from and to an AWS S3 bucket using
    boto3 library.

    Attributes:
        bucket (str): name of the s3 bucket
        filename (str): path and name of the file in the bucket
        access_key (str): name of the environment variable where access key is saved
        secret_key (str): name of the environment variable where secret key is saved
        load_params (dict, optional): Pandas API load params. Defaults to None.
        save_params (dict, optional): Pandas API save params. Defaults to None.
        description (str, optional): Description of the dataset. Defaults to None.

    Example:
        .. code-block:: python
        
            cars = S3CSVDataset(bucket="mybucket", 
                    filename='path/to/file.csv'
                    access_key="AWS_ACCESS_KEY", 
                    secret_key="AWS_SECRET_KEY"
                    )

    """
    def __init__(self, bucket:str, filename:str,
                access_key:str, secret_key:str,load_params:dict=None,
                save_params:dict=None, description:str=None
                ) -> None:

        if not filename.endswith('.csv'):
            self._logger.warning(f"{filename} does not ends with .csv. Modifying extention")
            filename = filename + ".csv"   

        super().__init__(bucket, filename,
                access_key, secret_key, description)

        self.load_params = load_params if load_params is not None else {}
        self.save_params = save_params if save_params is not None else {}


    def _load(self):
        response = self._get_s3_object()
        df = pd.read_csv(response['Body'], **self.load_params)
        return df

    def _save(self, data):
        out_buffer = io.StringIO()
        data.to_csv(out_buffer, **self.save_params)
        self._put_s3_object(out_buffer)   


class S3PickleDataset(_BaseS3Dataset):
    """Loads and Saves a pkl file from and to an AWS S3 bucket using
    boto3 library.

    Attributes:
        bucket (str): name of the s3 bucket
        filename (str): path and name of the file in the bucket
        access_key (str): name of the environment variable where access key is saved
        secret_key (str): name of the environment variable where secret key is saved
        description (str, optional): Description of the dataset. Defaults to None.

    Example:
        .. code-block:: python
        
            cars = S3PickleDataset(bucket="mybucket", 
                    filename='path/to/file.pkl'
                    access_key="AWS_ACCESS_KEY", 
                    secret_key="AWS_SECRET_KEY"
                    )

    """
    def __init__(self, bucket:str, filename:str,
                access_key:str, secret_key:str, description:str=None) -> None:
            
        if not filename.endswith('.pkl'):
            self._logger.warning(f"{filename} does not ends with .pkl. Modifying extention")
            filename = filename + ".pkl"
        super().__init__(bucket, filename,
                access_key, secret_key, description)

    def _load(self):
        response = self._get_s3_object()
        return pickle.load(io.BytesIO(response['Body'].read()))


    def _save(self, data):
        out_buffer = io.BytesIO()
        pickle.dump(data, out_buffer, pickle.HIGHEST_PROTOCOL)
        self._put_s3_object(out_buffer)



class S3JSONDataset(_BaseS3Dataset):
    """Loads and Saves a json file from and to an AWS S3 bucket using
    boto3 library.

    Attributes:
        bucket (str): name of the s3 bucket
        filename (str): path and name of the file in the bucket
        access_key (str): name of the environment variable where access key is saved
        secret_key (str): name of the environment variable where secret key is saved
        description (str, optional): Description of the dataset. Defaults to None.

    Example:
        .. code-block:: python
        
            cars = S3JSONDataset(bucket="mybucket", 
                    filename='path/to/file.json'
                    access_key="AWS_ACCESS_KEY", 
                    secret_key="AWS_SECRET_KEY"
                    )

    """
    def __init__(self, bucket:str, filename:str,
                access_key:str, secret_key:str, description:str=None) -> None:

        if not filename.endswith('.json'):
            self._logger.warning(f"{filename} does not ends with .json. Modifying extention")
            filename = filename + ".json"
            
        super().__init__(bucket, filename,
                access_key, secret_key, description)

    def _load(self):
        response = self._get_s3_object()
        return json.load(response['Body'])

    def _save(self, data):
        out_buffer = io.StringIO()
        json.dump(data,out_buffer)
        self._put_s3_object(out_buffer)




class S3YAMLDataset(_BaseS3Dataset):
    """Loads and Saves a yml file from and to an AWS S3 bucket using
    boto3 library.

    Attributes:
        bucket (str): name of the s3 bucket
        filename (str): path and name of the file in the bucket
        access_key (str): name of the environment variable where access key is saved
        secret_key (str): name of the environment variable where secret key is saved
        description (str, optional): Description of the dataset. Defaults to None.

    Example:
        .. code-block:: python
        
            cars = S3JSONDataset(bucket="mybucket", 
                    filename='path/to/file.yml'
                    access_key="AWS_ACCESS_KEY", 
                    secret_key="AWS_SECRET_KEY"
                    )

    """
    def __init__(self, bucket:str, filename:str,
                access_key:str, secret_key:str, description:str=None) -> None:

        if not filename.endswith('.yml'):
            self._logger.warning(f"{filename} does not ends with .yml. Modifying extention")
            filename = filename + ".yml"
            
        super().__init__(bucket, filename,
                access_key, secret_key, description)

    def _load(self):
        response = self._get_s3_object()
        return yaml.safe_load(response['Body'])

    def _save(self, data):
        out_buffer = io.StringIO()
        yaml.dump(data, out_buffer)
        self._put_s3_object(out_buffer)





