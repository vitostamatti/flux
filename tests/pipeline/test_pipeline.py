from flux.pipeline import Node, Pipeline

import pytest

def my_func_1(arg1:float, arg2:float):
    return arg1+arg2

@pytest.fixture
def basic_node_1():
    return Node(
        func = my_func_1,
        inputs = ['arg1',"arg2"],
        outputs = ["out_1"],
        name = "my_node_1"
    )

def my_func_2(arg1:float, arg2:float):
    return arg1/arg2

@pytest.fixture
def basic_node_2():
    return Node(
        func = my_func_2,
        inputs = ['arg1',"arg2"],
        outputs = ["out_2"],
        func_kwargs={"kwarg1":5},
        name = "my_node_2"
    )

def my_func_3(arg1:float, arg2:float):
    return arg1*arg2

@pytest.fixture
def basic_node_3():
    return Node(
        func = my_func_3,
        inputs = ['out_1','out_2'],
        outputs = ["out_3"],
        name = "my_node_3"
    )



class TestPipeline():

    def test_init(self, basic_node_1,basic_node_2, basic_node_3):
        pipe = Pipeline(nodes = [basic_node_1, basic_node_2, basic_node_3])

        assert pipe.all_inputs == {'arg1','arg2','out_1','out_2'}
        assert pipe.all_outputs  == {'out_1','out_2','out_3'}
        assert pipe.datasets == {'arg1','arg2','out_1','out_2','out_3'}
        assert pipe.intermediates_inputs  == {'out_1','out_2'}
        assert pipe.inputs == {'arg1','arg2'}
        assert pipe.outputs == {'out_3'}
        assert pipe.last_output == 'out_3'
        assert pipe.nodes == [basic_node_1, basic_node_2, basic_node_3]    
        assert pipe._exists(basic_node_1) == True

        nodes = [basic_node_1, basic_node_2, basic_node_3]
        assert pipe._nodes_by_name == {node.name:node for node in nodes}
        assert pipe._nodes_by_input == {
            "arg1":{basic_node_1, basic_node_2},
            "arg2":{basic_node_1, basic_node_2},
            "out_1":{basic_node_3},
            "out_2":{basic_node_3}
            }
        assert pipe._nodes_by_output == {
            "out_1":basic_node_1,
            "out_2":basic_node_2,
            "out_3":basic_node_3
            }
        assert pipe._topo_sorted_nodes == [basic_node_1, basic_node_2, basic_node_3]
        assert pipe.node_dependencies == {
            basic_node_1:set(), 
            basic_node_2:set(), 
            basic_node_3:{basic_node_1,basic_node_2}}
        

    def test_add_nodes(self, basic_node_1,basic_node_2, basic_node_3):
        pipe = Pipeline()
        pipe.add_nodes(node = basic_node_1)
        assert pipe._exists(basic_node_1) == True
        pipe.add_nodes(node = [basic_node_2,basic_node_3])
        assert (pipe._exists(basic_node_2) == True)
        assert (pipe._exists(basic_node_3) == True)

    def test_repr_str_add_eq(self, basic_node_1,basic_node_2):
        pipe = Pipeline(nodes=[basic_node_1])
        assert pipe.__repr__() == "Pipeline \n - Node: my_node_1([arg1,arg2]) -> [out_1]\n"
        assert pipe.__str__() == "Pipeline \n - Node: my_node_1([arg1,arg2]) -> [out_1]\n"

        pipe1 = Pipeline(nodes=[basic_node_1])
        pipe2 = Pipeline(nodes=[basic_node_2])
        pipe3 = pipe1+pipe2
        assert  pipe3._exists(basic_node_1) == True
        assert pipe3._exists(basic_node_2) == True
        assert pipe3 == pipe1+pipe2