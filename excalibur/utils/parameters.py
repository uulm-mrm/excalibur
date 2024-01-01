import copy


def add_default_kwargs(input_kwargs, **default_kwargs):
    # check input
    if input_kwargs is None:
        return default_kwargs

    # copy input
    input_kwargs = copy.copy(input_kwargs)

    # add default
    for k, v in default_kwargs.items():
        if k not in input_kwargs:
            input_kwargs[k] = v
        elif isinstance(input_kwargs[k], dict) and isinstance(v, dict):
            input_kwargs[k] = add_default_kwargs(input_kwargs[k], **v)

    return input_kwargs
