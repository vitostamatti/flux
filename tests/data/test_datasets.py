
from wrangler.data.datasets import (
    MemoryDataset,
    CSVDataset,
    ExcelDataset,
    ParquetDataset,
    S3CSVDataset,
    JSONDataset,
    PickleDataset,
    YAMLDataset
)
import pytest
import pandas as pd
import pickle
import yaml
import json


@pytest.fixture
def pandas_dataframe():
    return pd.DataFrame({"a":[1,2,3,4],"b":[1,2,3,4]})


class TestMemoryDataset():
    def test_load_python_obj(self):
        data = {1,2,3}
        ds = MemoryDataset(data)
        assert data == ds.load()

    def test_save_python_obj(self):
        data = {1,2,3}
        ds = MemoryDataset()
        ds.save(data)
        assert data == ds.data

    def test_load_pandas_dataframe(self, pandas_dataframe):
        ds = MemoryDataset(pandas_dataframe)
        assert pandas_dataframe.equals(ds.load())

    def test_save_pandas_dataframe(self, pandas_dataframe):
        ds = MemoryDataset()
        ds.save(pandas_dataframe)
        assert pandas_dataframe.equals(ds.data)


@pytest.fixture(scope="session")
def path_to_csv_file(tmp_path_factory):
    data = pd.DataFrame({"a":[1,2,3,4],"b":[1,2,3,4]})
    path = tmp_path_factory.mktemp("data") / "data.csv"
    data.to_csv(path, index=False)
    return path


class TestCSVDataset():
    def test_load(self, path_to_csv_file):
        # without params
        ds = CSVDataset(path_to_csv_file)
        data = pd.read_csv(path_to_csv_file)
        assert data.equals(ds.load())

        # with params
        ds = CSVDataset(path_to_csv_file, save_params={"index":False},load_params={"sep":","})
        data = pd.read_csv(path_to_csv_file)
        assert data.equals(ds.load())

    def test_save(self, path_to_csv_file, pandas_dataframe):
        # without params
        ds = CSVDataset(path_to_csv_file)
        ds.save(pandas_dataframe)
        assert pandas_dataframe.equals(ds.load().iloc[:,1:])

        # with params
        ds = CSVDataset(path_to_csv_file, save_params={"index":False}, load_params={"sep":","})
        ds.save(pandas_dataframe)
        assert pandas_dataframe.equals(ds.load())



@pytest.fixture(scope="session")
def path_to_excel_file(tmp_path_factory):
    data = pd.DataFrame({"a":[1,2,3,4],"b":[1,2,3,4]})
    path = tmp_path_factory.mktemp("data") / "data.xlsx"
    data.to_excel(path, index=False)
    return path


class TestExcelDataset():
    def test_load(self, path_to_excel_file):
        ds = ExcelDataset(path_to_excel_file)
        data = pd.read_excel(path_to_excel_file)
        assert data.equals(ds.load())

        ds = ExcelDataset(path_to_excel_file, save_params={"index":False}, load_params={"decimal":"."})
        data = pd.read_excel(path_to_excel_file)
        assert data.equals(ds.load())

    def test_save(self, path_to_excel_file, pandas_dataframe):
        ds = ExcelDataset(path_to_excel_file, save_params={"index":False})
        ds.save(pandas_dataframe)
        assert pandas_dataframe.equals(ds.load())

        ds = ExcelDataset(path_to_excel_file, save_params={"index":False}, load_params={"decimal":"."})
        ds.save(pandas_dataframe)
        assert pandas_dataframe.equals(ds.load())


@pytest.fixture(scope="session")
def path_to_parquet_file(tmp_path_factory):
    data = pd.DataFrame({"a":[1,2,3,4],"b":[1,2,3,4]})
    path = tmp_path_factory.mktemp("data") / "data.parquet"
    data.to_parquet(path, index=False)
    return path


class TestParquetDataset():
    def test_load(self, path_to_parquet_file):
        ds = ParquetDataset(path_to_parquet_file)
        data = pd.read_parquet(path_to_parquet_file)
        assert data.equals(ds.load())

    def test_save(self, path_to_parquet_file, pandas_dataframe):
        ds = ParquetDataset(path_to_parquet_file)
        ds.save(pandas_dataframe)
        assert pandas_dataframe.equals(ds.load())


from dotenv import load_dotenv

# class TestS3CSVDataset():

#     def test_load(self, pandas_dataframe):
#         load_dotenv()
#         ds = S3CSVDataset(
#             bucket="wrangler-platform", filename='tests/data.csv', 
#             access_key="AWS_ACCESS_KEY", secret_key="AWS_SECRET_KEY",
#             save_params={"index":False}
#             )
#         assert pandas_dataframe.equals(ds.load())

    # arreglar!
    # def test_save(self, pandas_dataframe):
    #     load_dotenv()
    #     ds = S3CSVDataset(
    #         bucket="wrangler-test", filename='tests/data.csv', 
    #         access_key="AWS_ACCESS_KEY", secret_key="AWS_SECRET_KEY",
    #         save_params={"index":False}
    #         )
    #     ds.save(pandas_dataframe)
    #     assert pandas_dataframe.equals(ds.load())




@pytest.fixture(scope="session")
def path_to_json_file(tmp_path_factory):
    data = {"a":[1,2,3,4],"b":[1,2,3,4]}
    path = tmp_path_factory.mktemp("data") / "data.json"
    with open(path,'w') as f:
        json.dump(data, f)
    return path

@pytest.fixture(scope="session")
def dict_data():
    data = {"a":[1,2,3,4],"b":[1,2,3,4]}
    return data


class TestJSONDataset():
    def test_load(self, path_to_json_file, dict_data):
        ds = JSONDataset(path_to_json_file)
        assert dict_data == ds.load()

    def test_save(self, path_to_json_file, dict_data):
        ds = JSONDataset(path_to_json_file)
        ds.save(dict_data)
        assert dict_data == ds.load()


@pytest.fixture(scope="session")
def path_to_pickle_file(tmp_path_factory):
    data = {"a":[1,2,3,4],"b":[1,2,3,4]}
    path = tmp_path_factory.mktemp("data") / "data.pkl"
    with open(path,'wb') as f:
        pickle.dump(data,f)
    return path


class TestPickleDataset():
    def test_load(self, path_to_pickle_file, dict_data):
        ds = PickleDataset(path_to_pickle_file)
        assert dict_data == ds.load()

    def test_save(self, path_to_pickle_file, dict_data):
        ds = PickleDataset(path_to_pickle_file)
        ds.save(dict_data)
        assert dict_data == ds.load()


@pytest.fixture(scope="session")
def path_to_yaml_file(tmp_path_factory):
    data = {"a":[1,2,3,4],"b":[1,2,3,4]}
    path = tmp_path_factory.mktemp("data") / "data.yml"
    with open(path,'w') as f:
        yaml.dump(data, f)
    return path


class TestYAMLDataset():
    def test_load(self, path_to_yaml_file, dict_data):
        ds = YAMLDataset(path_to_yaml_file)
        assert dict_data == ds.load()

    def test_save(self, path_to_yaml_file, dict_data):
        ds = YAMLDataset(path_to_yaml_file)
        ds.save(dict_data)
        assert dict_data == ds.load()
