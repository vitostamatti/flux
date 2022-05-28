from wrangler import Wrangler

def describe_wrangler(wrangler:Wrangler, output_path:str=None):
    import json
    import os
    datasets = []
    for ds in wrangler.pipeline.inputs():
        ds_params = {}
        for param, value in wrangler.data_catalog.datasets[ds].__dict__.items():
            if value is not None:
                ds_params[param] = str(value)
        datasets.append(ds_params)


    nodes = []
    for node in wrangler.pipeline.nodes:
        n_params = {}
        for param, value in node.__dict__.items():
            if value is not None:
                n_params[param] = str(value)
        nodes.append(n_params)

    data = {}
    data['datasets'] = datasets
    data['nodes'] = nodes

    if output_path is not None:
        with open(os.path.join(output_path,"wrangler.json"),'w') as file:
            json.dump(data, file)

    return data