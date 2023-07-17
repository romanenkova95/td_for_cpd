def replace_layer_by_name(model: torch.nn.Module, 
                          mname: str, 
                          layer: torch.nn.Module):
    """Insert layer into model according to path
    Args:
        model: original model
        mname: path to insert
        layer: layer to insert
    """
    module = model
    mname_list = mname.split('.')
    for mname in mname_list[:-1]:
        module = module._modules[mname]

    module._modules[mname_list[-1]] = layer

