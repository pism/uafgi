
def select_kwargs(kwargs0, defaults):
    """Selects only keys from kwargs0 that are also in defaults."""
    kwargs = dict(defaults.items())
    kwargs.update((k,v) for k,v in kwargs0.items() if k in defaults)
    return kwargs
