import pytest

from flux.extras.plot import plot_flux
from flux.data import DataCatalog
from flux.data.datasets import CSVDataset
from flux.core.flux import Wrangler
from flux.pipeline import Pipeline, Node



def _data_catalog():
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


def _pipeline():
    return Pipeline(
        nodes = [basic_node_1(), basic_node_2(), basic_node_3()]
    )


@pytest.fixture
def fixture_wrangler():
    return Wrangler(pipeline = _pipeline(), catalog=_data_catalog())


def test_plot_wrangler(fixture_wrangler):
    g = plot_wrangler(fixture_wrangler)
    assert g is not None