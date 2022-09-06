from flux.data.catalog import DataCatalog
from flux.data.datasets import (
    MemoryDataset
)
import pytest
import pandas as pd


@pytest.fixture
def memory_dataset():
    data = pd.DataFrame({"a":[1,2,3,4],"b":[1,2,3,4]})
    return MemoryDataset(data)

class TestDataCatalog():
    
    def test_add(self, memory_dataset):
        dc = DataCatalog()
        dc.add("test_dataset", memory_dataset)
        assert memory_dataset.load().equals(dc._datasets['test_dataset'].load())

    def test_register(self, memory_dataset):
        dc = DataCatalog()
        dc.register("test_dataset", memory_dataset)
        assert memory_dataset.load().equals(dc._datasets['test_dataset'].load())

    def test_release(self, memory_dataset):
        dc = DataCatalog()
        dc.add("test_dataset", memory_dataset)
        dc.release("test_dataset")
        assert dc.datasets.get("test_dataset", None)==None     

    def test_load(self, memory_dataset):
        dc = DataCatalog()
        dc.add("test_dataset", memory_dataset)
        loaded_data = dc.load("test_dataset")
        assert isinstance(loaded_data, dict)
        assert memory_dataset.load().equals(loaded_data["test_dataset"])

    def test_load_list(self,memory_dataset):
        dc = DataCatalog()
        dc.add("test_dataset_1", memory_dataset)
        dc.add("test_dataset_2", memory_dataset)        

        loaded_data = dc.load(["test_dataset_1","test_dataset_2"])

        assert isinstance(loaded_data, dict)
        assert "test_dataset_1" in loaded_data
        assert "test_dataset_2" in loaded_data


    def test_save_new_dataset(self):
        dc = DataCatalog()
        data = pd.DataFrame({"a":[1,2,3,4],"b":[1,2,3,4]})
        dc.save("test_dataset", data)
        assert data.equals(dc._datasets['test_dataset'].load()) 

    def test_save_existing_dataset(self,memory_dataset):
        dc = DataCatalog()
        data = pd.DataFrame({"a":[1,2,3,4],"b":[1,2,3,4]})
        dc.add("test_dataset", memory_dataset)
        dc.save("test_dataset", data)
        assert memory_dataset.load().equals(dc._datasets['test_dataset'].load()) 

    def test_repr_str(self,memory_dataset):
        dc = DataCatalog()
        dc.add("test_dataset", memory_dataset)
        
        assert dc.__repr__() == str({
            "test_dataset":memory_dataset
        })
        assert dc.__str__() == str({
            "test_dataset":memory_dataset
        })

        dc2 = DataCatalog()
        dc2.add("test_dataset", memory_dataset)
        assert dc == dc2