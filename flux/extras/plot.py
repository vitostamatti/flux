


def _plot_transform_inputs(flux, g):

    from graphviz import nohtml

    for i, inputs in enumerate(flux.pipeline.inputs):
        input_string = 'Data | { Name | Type} | {'
        input_name = str(inputs) + "|"
        if inputs in flux.catalog.datasets:
            input_type = str(flux.catalog.datasets.get(inputs)) + "}"
        else:
            input_type = "MemoryDataset  }"
        input_string = input_string + input_name + input_type

        g.node(f'node_data_i_{i}', nohtml(input_string))


    for i, outputs in enumerate(flux.pipeline.outputs):
        output_string = 'Data | { Name | Type} | {'
        output_name = str(outputs) + "|"
        if outputs in flux.catalog.datasets:
            output_type = str(flux.catalog.datasets.get(outputs)) + "}"
        else:
            output_type = "PandasDataset  }"
        output_string = output_string + output_name + output_type
        g.node(f'node_data_o_{i}', nohtml(output_string))


    for i, node in enumerate(flux.pipeline.nodes):
        node_string = 'Node | {Name | Func |Inputs | Outputs} | {'
        node_name = str(node.name) + "|"
        node_func = str(node.func) + "|"
        node_inputs = str(node.inputs) + "|"
        node_outputs = str(node.outputs) + "}"
        node_string = node_string + node_name + node_func + node_inputs + node_outputs
        g.node(
            f"node{i}",
            nohtml(node_string)
        )

    for i, inputs in enumerate(flux.pipeline.inputs):
        for j, node in enumerate(flux.pipeline.nodes):
            if inputs in node.inputs:
                g.edge(f'node_data_i_{i}', f'node{j}')  

    nodes = flux.pipeline.nodes
    adj = []
    for i in range(len(nodes)-1, 0 , -1):
        to_node = nodes[i]
        for to_input in to_node.inputs:
            for j in range(i, 0, -1):
                if (f'node{j-1}', f'node{i}') in adj:
                    break
                elif to_input in nodes[j-1].outputs:
                    adj.append((f'node{j-1}', f'node{i}'))
                    g.edge(f'node{j-1}', f'node{i}')
                    break

    for i, outputs in enumerate(flux.pipeline.outputs):
        for j in range(len(flux.pipeline.nodes)-1, 0 , -1):
            if outputs in flux.pipeline.nodes[j].outputs:
                g.edge(f'node{j}', f'node_data_o_{i}')  
                break
    return g


def plot_flux(flux, filepath:str=None):
    """Util function to generate a graph plot of the flux

    Args:
        flux (Flux): the flux to plot. The fit_mode of
            will afect the resulting plot.
        filepath (str, optional): optional path to save plot. Defaults to None.

    Returns:
        graphviz.Digraph: graph of object of the flux.
    """
    import graphviz
    from graphviz import nohtml


    g = graphviz.Digraph('dspl',
        node_attr={'shape': 'record', 'height': '.1'}, 
        format='png', 
        graph_attr={'size' : "18,18!"}
    )
    
    g = _plot_transform_inputs(flux, g)

    if filepath is not None:
        g.view(filepath)
    return g
