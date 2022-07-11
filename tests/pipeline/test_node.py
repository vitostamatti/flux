from wrangler.pipeline import Node

import pytest

def my_func(arg1, arg2, kwarg1=2):
    return arg1+arg2+kwarg1

def my_func_2(arg):
    return arg


@pytest.fixture
def basic_node():
    return Node(
        func = my_func,
        inputs = ['arg1',"arg2"],
        outputs = ["sum_output"],
        func_kwargs={"kwarg1":5},
        name = "my_node"
    )

@pytest.fixture
def basic_node_2():
    return Node(
        func = my_func,
        inputs = ['arg1',"arg2"],
        outputs = ["sum_output"],
        func_kwargs={"kwarg1":5},
        name = "my_node"
    )

@pytest.fixture
def basic_node_3():
    return Node(
        func = my_func_2,
        inputs = ['sum_output'],
        outputs = ["same_output"],
        name = "my_node_3"
    )



class TestNode():

    def test_init(self, basic_node):
        assert basic_node.name == 'my_node'
        assert basic_node.inputs == ['arg1',"arg2"]
        assert basic_node.outputs == ['sum_output']
        assert basic_node.func_kwargs=={"kwarg1":5}
        assert basic_node._defaut_name == f"{my_func.__name__}"
        assert basic_node.func == my_func

    def test_run(self, basic_node):
        outputs = basic_node.run(
            {"arg1":10,"arg2":10}
        )
        assert outputs == {"sum_output":25}

    def test_repr_str(self, basic_node):
        assert basic_node.__repr__() == "Node: my_node([arg1,arg2]) -> [sum_output]\n"
        assert basic_node.__str__() == "Node: my_node([arg1,arg2]) -> [sum_output]\n"

    def test_eq(self, basic_node, basic_node_2):
        assert basic_node==basic_node_2

    def test_lt(self, basic_node, basic_node_3):
        assert basic_node<basic_node_3