import io


def sprint(*args, **kwargs):
    """print to string"""
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents


def map_interface_args(paths, *args, **kwargs):
    """Map arguments for interface if they are not lists"""
    n = len(paths)
    for i, path in enumerate(paths):
        targs = []
        for arg in args:
            if not (isinstance(arg, list)) or len(arg) != n:
                targs.append(arg)
            else:
                targs.append(arg[i])
        tkwargs = {}
        for k, v in kwargs.items():
            if not (isinstance(v, list)) or len(v) != n:
                tkwargs[k] = v
            else:
                tkwargs[k] = v[i]
        yield path, targs, tkwargs
