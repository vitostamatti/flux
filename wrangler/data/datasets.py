# Implementations of Dataset

from loguru import logger
import pandas as pd
from wrangler.base import AbstractDataset
from typing import Any, AnyStr, List
import io
import os
# import boto3


class PandasDataset(AbstractDataset):
    """In Memory datset with a Pandas DataFrame asociated.

    Attributes:
        data (pd.DataFrame, optional): [description]. Defaults to None.

    Example:
        .. code-block:: python

            cars_data = pd.DataFrame({
                "cars":["car1", "car2", "car3", "car4"],
            })
            cars = PandasDataset(data=cars_data)

    """

    def __init__(self, name:str, data:pd.DataFrame=None,
                description:str=None ) -> None:

        super().__init__(name, description)
        self.data = data

    def _load(self) -> Any:
        if self.data is None:
            raise Exception("This dataset does not have any data.")
        data = self.data.copy()
        return data

    def _save(self, data: Any):
        self.data = data


class CSVDataset(AbstractDataset):
    """Loads and Saves a .csv file from and to the local filesystem
    with specific parameters usign Pandas API.

    Attributes:
        filename (str): [description]
        load_params (dict, optional): [description]. Defaults to None.
        save_params (dict, optional): [description]. Defaults to None.

    Example:
        .. code-block:: python

            cars = PandasDataset(filename='path/to/file.csv')

    """
    def __init__(self, name:str, filename:str,
                description:str=None ,load_params:dict=None,
                save_params:dict=None) -> None:

        super().__init__(name, description)
        self.filename = filename
        self.load_params = load_params
        self.save_params = save_params

    def _load(self):
        if self.load_params is not None:
            return pd.read_csv(self.filename, **self.load_params)
        else:
            return pd.read_csv(self.filename)

    def _save(self, data):
        if isinstance(data, pd.DataFrame):
            if self.save_params is not None:
                data.to_csv(self.filename, **self.save_params)
            else:
                data.to_csv(self.filename)


class ParquetDataset(AbstractDataset):
    """Loads and Saves a .parquet file from and to the local filesystem
    with specific parameters usign Pandas API.

    Attributes:
        filename (str): [description]
        load_params (dict, optional): [description]. Defaults to None.
        save_params (dict, optional): [description]. Defaults to None.

    Example:
        .. code-block:: python

            cars = ParquetDataset(filename='path/to/file.parquet')

    """

    def __init__(self, name: str, filename: str,
                 description: str = None, load_params: dict = None,
                 save_params: dict = None) -> None:

        super().__init__(name, description)
        self.filename = filename
        self.load_params = load_params
        self.save_params = save_params

    def _load(self):
        if self.load_params is not None:
            return pd.read_parquet(self.filename, **self.load_params)
        else:
            return pd.read_parquet(self.filename)

    def _save(self, data):
        if isinstance(data, pd.DataFrame):
            if self.save_params is not None:
                data.to_parquet(self.filename, **self.save_params)
            else:
                data.to_parquet(self.filename)


class ExcelDataset(AbstractDataset):
    """Loads and Saves an Excel file from and to the local filesystem
    with specific parameters usign Pandas API.

    Attributes:
        filename (str): [description]
        load_params (dict, optional): [description]. Defaults to None.
        save_params (dict, optional): [description]. Defaults to None.

    Example:
        .. code-block:: python

            cars = PandasDataset(filename='path/to/file.xlsx')

    """
    def __init__(self, name:str ,filename:str,
                load_params:dict=None, save_params:dict=None,
                description:str=None) -> None:

        super().__init__(name, description)
        self.load_params = load_params
        self.save_params = save_params
        self.filename = filename

    def _load(self):
        if self.load_params is not None:
            return pd.read_excel(self.filename, **self.load_params)
        else:
            return pd.read_excel(self.filename)

    def _save(self, data):
        if isinstance(data, pd.DataFrame):
            if self.save_params is not None:
                data.to_excel(self.filename, **self.save_params)
            else:
                data.to_excel(self.filename)


class S3Dataset(AbstractDataset):
    """Loads and Saves files from an s3 bucket with a given authenticated client

    It suports Excel, Comma Separated Values and Parquet file formats.

    Attributes:
        bucket (str): the bucket name to load and save.
        filename (str): the path to the file. Suported file formats: csv, parquet, xlsx.
        s3_client ([type]): an instance of a s3 client brom boto3 library.
        load_params (dict, optional): all keyword arguments of the pd.load_[fileformat]
            method from pandas. Defaults to None.
        save_params (dict, optional): all keyword arguments of the DataFrame.to_[fileformat]
            method from pandas. Defaults to None.
    """
    def __init__(self, name:str ,bucket:str, filename:str,
                access_key, secret_key, load_params:dict=None,
                save_params:dict=None, description:str=None) -> None:

        super().__init__(name, description)
        self.bucket = bucket
        self.filename = filename

        self.access_key = access_key
        self.secret_key = secret_key

        self.load_params = load_params if load_params is not None else {}
        self.save_params = save_params if save_params is not None else {}


    def _get_file_from_s3(self):
        s3_client = self._get_s3_client()

        response = s3_client.get_object(Bucket=self.bucket, Key=self.filename)
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status == 200:
            self._logger.info(f"Successful S3 get_object response. Status - {status}")
        else:
            raise Exception(f"Unsuccessful S3 get_object response. Status - {status}")

        if self.filename.split(".")[-1] == 'csv':
            df = pd.read_csv(response['Body'], **self.load_params)

        elif self.filename.split(".")[-1] == 'parquet':
            df = pd.read_parquet(io.BytesIO(response['Body'].read()), **self.load_params)

        elif self.filename.split(".")[-1] == 'xlsx':
            df = pd.read_excel(io.BytesIO(response['Body'].read()), **self.load_params)

        return df

    def _put_file_to_s3(self, data):
        s3_client = self._get_s3_client()

        if self.filename.split(".")[-1] == 'parquet':
            out_buffer = io.BytesIO()
            data.to_parquet(out_buffer, index=False, **self.save_params)

        elif self.filename.split(".")[-1] == 'csv':
            out_buffer = io.StringIO()
            data.to_csv(out_buffer, **self.save_params)

        elif self.filename.split(".")[-1] == 'xlsx':
            out_buffer = io.BytesIO()
            data.to_excel(out_buffer, **self.save_params)
        else:
            raise Exception(f"File format {self.fileformat} not supported.")

        response = s3_client.put_object(
                Bucket=self.bucket, Key=self.filename, Body=out_buffer.getvalue()
            )

        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status == 200:
            self._logger.info(f"Successful S3 put_object response. Status - {status}")
        else:
            raise Exception(f"Unsuccessful S3 put_object response. Status - {status}")

    def _get_s3_client(self):
        import boto3
        s3_client = boto3.client(
            service_name='s3',
            region_name='us-east-2',
            aws_access_key_id=os.environ.get(self.access_key),
            aws_secret_access_key=os.environ.get(self.secret_key),
        )
        return s3_client


    def _load(self):
        return self._get_file_from_s3()

    def _save(self, data):
        self._put_file_to_s3(data)


class SparkDataset(AbstractDataset):
    def __init__(self) -> None:
        pass

    def _load(self):
        pass

    def _save(self):
        pass
