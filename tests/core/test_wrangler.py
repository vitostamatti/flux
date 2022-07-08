from distutils.dep_util import newer_pairwise
from tkinter import W
import pytest
from wrangler.data.catalog import DataCatalog
from wrangler.pipeline import Node, Pipeline
from wrangler.core.wrangler import Wrangler

from wrangler.data.catalog import DataCatalog
from wrangler.data.datasets import (
    MemoryDataset,
    CSVDataset
)
import pandas as pd
import os


@pytest.fixture
def basic_data_catalog():
    dc = DataCatalog()
    dc.add("arg1", MemoryDataset(data=10))
    dc.add("arg2", MemoryDataset(data=20))
    return dc

@pytest.fixture
def basic_data_catalog_2():
    dc = DataCatalog()
    dc.add("arg1", CSVDataset(filename="arg1.csv"))
    dc.add("arg2", CSVDataset(filename="arg2.csv"))
    dc.add("out_3", CSVDataset(filename="out_3.csv"))    
    return dc


def my_func_1(arg1:float, arg2:float):
    return arg1+arg2


def basic_node_1():
    return Node(
        func = my_func_1,
        inputs = ['arg1',"arg2"],
        outputs = ["out_1"],
        name = "my_node_1"
    )

def my_func_2(arg1:float, arg2:float):
    return arg1-arg2


def basic_node_2():
    return Node(
        func = my_func_2,
        inputs = ['arg1',"arg2"],
        outputs = ["out_2"],
        name = "my_node_2"
    )

def my_func_3(arg1:float, arg2:float):
    return arg1*arg2


def basic_node_3():
    return Node(
        func = my_func_3,
        inputs = ['out_1','out_2'],
        outputs = ["out_3"],
        name = "my_node_3"
    )

@pytest.fixture
def basic_pipeline():
    return Pipeline(
        nodes = [basic_node_1(), basic_node_2(), basic_node_3()]
    )


@pytest.fixture
def fixture_basic_node_1():
    return basic_node_1()

@pytest.fixture
def fixture_basic_node_2():
    return basic_node_2()




class TestWrangler:

    def test_init(self):
        wrangler = Wrangler()
        assert isinstance(wrangler, Wrangler)
        assert isinstance(wrangler.pipeline, Pipeline)
        assert isinstance(wrangler.catalog, DataCatalog)

    def test_init_with_pipeline(self, basic_pipeline):
        wrangler = Wrangler(pipeline=basic_pipeline)
        assert wrangler.pipeline == basic_pipeline

    def test_init_with_catalog(self, basic_data_catalog):
        wrangler = Wrangler(catalog=basic_data_catalog)
        assert wrangler.catalog == basic_data_catalog

    def test_add_dataset(self):
        wrangler = Wrangler()
        data = 10
        ds = MemoryDataset(data=data)
        wrangler.add_dataset(name='dataset', dataset=ds)
        assert "dataset" in wrangler.catalog.datasets
        assert wrangler.catalog.load(names="dataset") == {"dataset":data}

        wrangler = Wrangler()
        ds = CSVDataset(filename="file.csv")
        wrangler.add_dataset(name='dataset', dataset=ds)
        assert "dataset" in wrangler.catalog.datasets


    def test_add_datasets(self):
        wrangler = Wrangler()
        data = 10
        ds = MemoryDataset(data=data)
        ds1 = MemoryDataset(data=data)
        wrangler.add_datasets({"dataset":ds,"dataset1":ds1})    
        assert "dataset" in wrangler.catalog.datasets
        assert "dataset1" in wrangler.catalog.datasets
        assert wrangler.catalog.load(names="dataset") == {"dataset":data}
        assert wrangler.catalog.load(names="dataset1") == {"dataset1":data}


    def test_add_node(self):
        wrangler = Wrangler()

        def _my_func(input):
            return input

        wrangler.add_node(
            func=_my_func,
            inputs=['input'],
            outputs=['output'],
            name="mynode"
        )

        assert len(wrangler.pipeline.nodes)==1

    def test_add_nodes(self, fixture_basic_node_1, fixture_basic_node_2):
        wrangler = Wrangler()

        wrangler.add_nodes(nodes=fixture_basic_node_1)

        assert len(wrangler.pipeline.nodes)==1

        wrangler = Wrangler()
        wrangler.add_nodes(nodes=[fixture_basic_node_1,fixture_basic_node_2])
        assert len(wrangler.pipeline.nodes)==2


    def test_run(self, basic_data_catalog, basic_pipeline):
        wrangler = Wrangler(pipeline=basic_pipeline, catalog=basic_data_catalog)
        wrangler.run()
        assert wrangler.catalog.load("out_3") == {"out_3": 30*-10}

        wrangler = Wrangler(pipeline=basic_pipeline, catalog=basic_data_catalog)
        wrangler()
        assert wrangler.catalog.load("out_3") == {"out_3": 30*-10}

    def test_save(self, tmpdir, basic_pipeline, basic_data_catalog):
        wrangler = Wrangler(pipeline=basic_pipeline, catalog=basic_data_catalog)
        wrangler.run()
        file = tmpdir.join('wrangler.pkl')
        wrangler.save(file.strpath)  # or use str(file)
        assert os.path.exists(file)

    def test_load(self, tmpdir, basic_pipeline, basic_data_catalog):
        wrangler = Wrangler(pipeline=basic_pipeline, catalog=basic_data_catalog)
        file = tmpdir.join('wrangler.pkl')
        wrangler.save(file.strpath)  # or use str(file)
        
        new_wrangler = Wrangler()
        new_wrangler.load(file.strpath)
        assert new_wrangler.pipeline == basic_pipeline 
        # MemoryDataset are not saved
        assert new_wrangler.catalog.datasets == {}

    def test_datasets_to_config(self, tmpdir, basic_pipeline, basic_data_catalog_2):
        wrangler = Wrangler(pipeline=basic_pipeline, catalog=basic_data_catalog_2)
        file = tmpdir.join('datasets.yml')
        wrangler.datasets_to_config(file.strpath)
        assert os.path.exists(file)


    def test_dataset_from_config(self, tmpdir, basic_pipeline, basic_data_catalog_2):
        wrangler = Wrangler(pipeline = basic_pipeline, catalog=basic_data_catalog_2)
        file = tmpdir.join('datasets.yml')
        wrangler.datasets_to_config(file.strpath)

        new_wrangler = Wrangler()
        new_wrangler.datasets_from_config(file.strpath)
        assert new_wrangler.catalog == basic_data_catalog_2