import logging
from typing import Union, AnyStr, List, Dict, Callable, Any
from collections import Counter
from itertools import chain
import dill

from wrangler.data.datasets import MemoryDataset, AbstractDataset
from wrangler.data.catalog import DataCatalog
from wrangler.pipeline.pipeline import Pipeline
from wrangler.pipeline.pipeline import Node


from wrangler.utils.config import write_config, read_config, load_dataset_object
from wrangler.utils.utils import _get_notebook_imports


from wrangler.utils.utils import _isnotebook, _getfile
from IPython.core.magics.code import extract_symbols
import inspect


class Wrangler(object):
    """
    A class to orchestrate and run a DataCatalog and a Pipeline of Nodes.

    Attributes:
        catalog (DataCatalog, optional): The catalog from which Wrangler will load and save
            all data. Defaults to None. If None, creates a default one with no data in it.
        pipeline (Pipeline, optional): The sequence of nodes which Wrangler will apply to the given
            datasets. Defaults to None. If None, creates a default one with no Nodes in it.

    Example:
        .. code-block:: python

            # Add example

    """

    _default_dataset_name = "default_dataset"

    def __init__(
            self, 
            catalog:DataCatalog = None, 
            pipeline:Pipeline = None
        ) -> None:

        self._catalog = DataCatalog() if catalog is None else catalog
        self._pipeline = Pipeline() if pipeline is None else pipeline

        self._source_code = {}

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    @property
    def pipeline(self)->Pipeline:
        """The Pipeline object inside.
        """
        return self._pipeline

    @property
    def catalog(self)->DataCatalog:
        """The DataCatalog object inside.
        """
        return self._catalog

    @property
    def source_code(self)->Dict:
        return self._source_code


    def load_dataset(self, name:str):
        """Load and return a dataset with the given name

        Args:
            name (str): name of the dataset

        Returns:
            Any: the data loaded from the dataset
        """
        return self.catalog.load(name)[name]


    def add_dataset(self, name: str, dataset: AbstractDataset)->None:
        """Register the given dataset with the given name to the data catalog
        of the wrangler.

        Args:
            dataset (AbstractDataset): an instance of an implementation of AbstractDataset.
        """
        if isinstance(dataset, AbstractDataset):
            self._catalog.add(name, dataset)
        else:
            msg = f"{dataset} must be a subclass of `AbstractDataset`"
            self._logger.error(msg)
            raise ValueError(msg)


    def add_datasets(self, datasets: Dict[str, AbstractDataset])->None:
        """Save a dictionary of datasets to the catalog

        Args:
            datasets (Dict) : a dictionary of {name:dataset}.

        Raises:
            ValueError: if any of the elements in the dict is not an instance of `AbstractDataset`.
            ValueError: if the the datasets are not provided in dict format.
        """
        if not isinstance(datasets, dict):
            raise ValueError(
                "datasets must be passed as a dictionary of {'name':AbstractDataset}"
            )
        for d_name in datasets:
            if isinstance(datasets[d_name], AbstractDataset):
                self._catalog.add(d_name, datasets[d_name])
            else:
                msg = f"{datasets[d_name]} must be a subclass of `AbstractDataset`"
                self._logger.error(msg)
                raise ValueError(msg)

    def add_node(
            self,
            func: Callable,
            inputs: Union[AnyStr, List],
            outputs: Union[AnyStr, List],
            func_kwargs: Union[Dict, None] = None,
            name: AnyStr = None,
        )->None:

        """
        Inserts a new node to the pipeline or overrides an
        existing node with the given name.

        Args:
            transformer (AbstractTransformer): an instance of an implementation of AbstractTransformer
                with its corresponding constructor parameters.
            inputs (Union[str,List[str]], optional): the names of the datasets already registered
                to use as inputs. Defaults to None.
            outputs (Union[str,List[str]], optional): the names of the datasets to use as outputs. Defaults to None.
                If the outputs already exists, it overrides the data, otherwise, it creates a new dataset.
            name (str, optional): the name of the node and also identifier. Defaults to None.
        """
        node = Node(
            func, inputs, outputs, func_kwargs, name
        )
        self._pipeline.add_nodes(node)

    def add_nodes(self, nodes: Union[Node, List[Node]])->None:
        """
        Insert a list of nodes in the pipeline

        Args:
            nodes (Union[Node, List[Node]]): list of nodes to insert in pipeline

        Raises:
            ValueError: If any of the elements in the list is not a `Node` instance.
            ValueError: If the element passed in nodes parameter is not a `Node` instance.
        """
        if isinstance(nodes, list):
            for n in nodes:
                if not isinstance(n, Node):
                    msg =  "All elements in nodes must be an instance of `Node`"
                    self._logger.error(msg)
                    raise ValueError(msg)
        else:
            if not isinstance(nodes, Node):
                msg = "nodes argument must be an instance `Node` or a list of Node instances"
                self._logger.error(msg)
                raise ValueError(
                    msg
                )

        self._pipeline.add_nodes(nodes)



    def save(self, path: str)->None:
        """
        It saves the catalog references of all datasets that are not
        in memory (MemoryDataset).

        It saves the sequence of nodes in the current pipeline.

        Args:
            path (str): destination path to save the object.
        """
        datasets = {}
        for name, dataset in self._catalog.datasets.items():
            if not isinstance(dataset, MemoryDataset):
                datasets[name] = dataset

        objs = {
            "catalog": DataCatalog(datasets=datasets),
            "pipeline": self._pipeline,
            "source_code":self._source_code
        }
        if not path.endswith(".pkl"):
            path = path+".pkl"
        with open(path, "wb") as f:
            dill.dump(objs, f)



    def load(self, path: str):
        """
        It loads the object from the given path and sets
        data catalog and pipeline attributes to the loaded ones.

        Args:
            path (str): origin path to load the object.
        """

        if not path.endswith(".pkl"):
            path = path+".pkl"

        with open(path, "rb") as f:
            objs = dill.load(f)

        self._pipeline = objs["pipeline"]
        self._catalog = objs["catalog"]
        self._source_code = objs["source_code"]



    def run(self):
        """Excecute all nodes in sequential order with the inputs
        defined in the catalog.
        """
        return self._run()


    def datasets_to_config(self, path:str):
        """
        Generate a .yml file with the catalog information.

        The resulting file has two main keys: inputs and outputs.
        
        Inside each key, is are all the datasets with is parameters.

        Args:
            path (str): path to which save the .yml file. Does not need to finish with .yml

        Returns:
            Wrangler: the same instance of the Wrangler
        """
        data = {}
        datasets_inputs = self._get_datasets_params(self._pipeline.inputs)
        datasets_outputs = self._get_datasets_params(self._pipeline.outputs)

        data["inputs"] = datasets_inputs
        data["outputs"] = datasets_outputs

        write_config(path, data)

        return self
    

    def datasets_from_config(self, path:str):
        """Load the catalog information from a pre-existing configuration file.

        Args:
            path (str): path to which load the .yml file. Does not need to finish with .yml

        Returns:
            Wrangler: the same instance of the Wrangler
        """
        config_data = read_config(path)

        data = {
            **config_data["inputs"],
            **config_data["outputs"],
        }

        for dataset_name in data:
            dataset_conf = data[dataset_name]

            obj_type = dataset_conf.pop("type")

            dataset_class = load_dataset_object(obj_type)
            dataset = dataset_class(**dataset_conf)
            self.add_dataset(dataset_name, dataset)

        return self


    def _create_default_dataset(self):
        """Creates a default dataset of type `MemoryDataset`"""
        default_dataset = MemoryDataset()
        self.add_dataset(self._default_dataset_name, default_dataset)


    def _run_node(self, node: Node, node_inputs: List):
        """Excecutes the node.run() method with its inputs and save the outputs"""
        inputs = self._catalog.load(node_inputs)
        outputs = node.run(inputs)
        self._save_node_outputs(outputs, self._catalog)


    def _run(self):
        """Run all nodes of the pipeline"""
        self._validate_inputs_datasets()

        nodes = self._pipeline.nodes
        done_nodes = set()

        load_counts = Counter(chain.from_iterable(n.inputs for n in nodes))
        pipeline_inputs = self._pipeline.inputs

        for node in self._pipeline.nodes:

            node_inputs = node.inputs

            self._run_node(node, node_inputs)

            done_nodes.add(node)

            for data_set in node_inputs:
                load_counts[data_set] -= 1
                if load_counts[data_set] < 1 and data_set not in pipeline_inputs:
                    self.catalog.release(data_set)

            for data_set in node.outputs:
                if load_counts[data_set] < 1 and data_set not in self._pipeline.outputs:
                    self.catalog.release(data_set)


    def _save_node_outputs(self, outputs: Dict[str, Any], catalog: DataCatalog):
        """Save outputs to the catalog"""
        for output_name in outputs:
            catalog.save(output_name, outputs[output_name])


    def _get_datasets_params(self, list_of_datasets):
        """Get all the parameters of the datasets for the configuration file"""
        datasets = {}
        for ds_name in list_of_datasets:
            if ds_name in self._catalog.datasets:
                ds_params = {}
                ds_params["type"] = type(self._catalog.datasets[ds_name]).__name__
                if ds_params["type"] == "MemoryDataset":
                    ds_params["data"] = "None"
                else:
                    for param, value in self._catalog.datasets[
                        ds_name
                    ].__dict__.items():
                        if param == "name":
                            continue
                        elif value is not None:
                            if isinstance(value, dict):
                                ds_params[param] = value
                            else:
                                ds_params[param] = str(value)
                datasets[ds_name] = ds_params
            else:
                ds_params = {}
                ds_params["type"] = "MemoryDataset"
                datasets[ds_name] = ds_params

        return datasets


    def _validate_inputs_datasets(self):
        """Check that pipeline inputs are available"""
        inputs = self._pipeline.inputs
        for i in inputs:
            if i not in self.catalog.datasets:
                msg = f"DataCatalog does not have required input {i} to execute."
                self._logger.error(msg)
                raise Exception(
                    msg
                )

    def _register_source_code(self, name, sc):
        self._source_code[name]=sc


    def export_source_code(self, input_file:str , output_file:str='wrangler_util.py'):
        """Utility function to generate .py file with all the @export_function
        declared in the current environment.

        This function us particularly useful if you want to run the wrangler
        in a new fresh environment where you don't have all the dependent
        functions and libraries imported.

        Example:
            .. code-block:: python
            
                from wrangler import Wrangler
                from wrangler import export_function
                import numpy as np
 
                wrangler = Wrangler()

                @export_function(wrangler=wrangler)
                def my_func_to_be_exported(arg1,arg2):
                    return arg1+arg2

                my_func_to_be_exported(arg1=10,arg2=10)

                # pre-declared wrangler
                wrangler.export_source_code(input_file='notebook.ipynb',output_file='wrangler.py')

                # will generate an output file with all the @export_function decorated
                # functions and the import statements in the current notebook
                # the output file would look like this

                from wrangler import Wrangler
                from wrangler import export_function
                import numpy as np 

                def my_func_to_be_exported(arg1,arg2):           
                    return arg1+arg2


        Args:
            input_file (str): input notebook file from which running wrangler
            output_file (str, optional): output file in .py format. Defaults to 'wrangler_util.py'.
        """

        if not input_file.endswith('ipynb'):
            raise Exception("input_file must be a .ipynb file")
        if not output_file.endswith(".py"):
            raise Exception("output_file must be a .py file")

        import_lines = _get_notebook_imports(input_file)
        with open(output_file,'w') as f:
            f.write(import_lines)
            f.write("\n")
            for sc in self._source_code:
                f.write("\n")
                f.write(self._source_code[sc])
                f.write("\n")


    def __call__(self):
        """Alternative to the run() method"""
        return self.run()


   
   
class export_function(object):
    """useful decorator to export functions that
    are nested inside other functions.

    Ones wrangler is fitted and working on a notebook environment,
    if you need to use it in a new environment, you'll also need
    all imported and used functions. This decorator automate that
    task.

    Args:
        object (Wrangler): wrangler to which export function
    """
    def __init__(self, wrangler:Wrangler):
        if not _isnotebook():
            raise Exception("export_function decorator can only be used in jupyter notebooks")
        self._wrangler = wrangler

    def _save_source_code(self, f):
        cell_code = "".join(inspect.linecache.getlines(_getfile(f)))
        sc = extract_symbols(cell_code, f.__name__)[0][0]
        self._wrangler._register_source_code(f.__name__,sc)

    def __call__(self, f):
        self._save_source_code(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapper


