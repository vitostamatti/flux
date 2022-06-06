## TODO LIST:
- cambiar nombre wrangler (liberia y repositorio): datascience-pipelines (repo) - dspl (lib)
- limpiar extras/ -> archivos viejos
- PandasDataset -> MemoryDataset
- PickleDataset -> to save .pickle
- JsonDataset -> to save .json
- StatelessTransformer (DataframeTransformer) -> Instancio con funciones
- StatefulTransformer (ModelTransformer) -> si o si herencia o sklearn model?
- Nodo con parametros: name, fit_inputs, transform_inputs, outputs, fit_params, transform_params


class Node():
    def __init__(
        self,
        name:str,
        transformer:Union[StatelessTransformer,StatefulTransformer,Callable],
        fit_inputs:Union[str,List,None],
        transform_inputs:Union[str,List],
        outputs:Union[str,List],
        fit_kwargs:Union[dict,None],
        transform_kwargs:Union[dict,None]
        ):

        self.name_ = name

        if isinstance(transformer,callable):
            self.transformer_ = StatelessTransformer(func=transformer)
        else:
            self.transformer_ = transformer

        if isinstance(fit_inputs,str):
            self.fit_inputs_ = [fit_inputs]
        else:
            self.fit_inputs_ = fit_inputs

        if isinstance(transform_inputs,str):
            self.transform_inputs_ = [transform_inputs]
        else:
            self.transform_inputs_ = transform_inputs

        if isinstance(outputs,str):
            self.outputs_ = [outputs]
        else:
            self.outputs_ = outputs

        self.fit_kwargs_ = fit_kwargs
        
        self.transform_kwargs_ = transform_kwargs

Node(
    name:str,
    transformer:Union[StatelessTransformer,StatefulTransformer,Callable],
    transform_inputs:Union[str,List],
    outputs:Union[str,List],
    fit_inputs:Union[str,List,None],
    fit_kwargs:Union[dict,None],
    transform_kwargs:Union[dict,None]
)

class StatelessTransformer(Transformer):
    def __init__(self, func):
        self.func_ = func

    def transform(self, *inputs,**transform_kwargs):
        self.func_(*inputs,**transform_kwargs)
    



StatelessTransformer(
    func:Callable
    func_kwargs:Union[dict,None]
)



StatefulTransformer:
    - Inheritance
    - Sklearn like object.


class StatefulTransformer(Transformer):
    def __init__(self, obj):
        self.obj_ = obj

    def fit(self, *fit_inputs, **fit_kwargs):
        self.obj_.fit(*fit_inputs, **fit_kwargs)
        return self

    def transform(self, *transform_inputs, **transform_kwargs):
        return self.obj_.transform(*transform_inputs, **transform_kwargs)

Usage:

StatefulTransformer(
    obj=None
)

class MyTransformer(StatefulTransformer):
    def __init__(self, params):
        self.params_ = params
    
    def fit(self, *fit_inputs, **fit_kwargs):
        pass

    def transform(self, *transform_inputs, **transform_kwargs):
        pass

