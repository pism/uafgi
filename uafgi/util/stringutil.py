import string

# https://stackoverflow.com/questions/17215400/format-string-unused-named-arguments
class PartialFormatter(string.Formatter):
    def __init__(self, default='{{{0}}}'):
        self.default=default

    def get_value(self, key, args, kwargs):
        if isinstance(key, int):
            ret = args[key]
        else:
            try:
                ret = kwargs[key]
            except KeyError:
                ret = '{'+key+'}'

        return ret

def partial_format(template, *args, **kwargs):
    fmt = PartialFormatter()
    return fmt.format(template, *args, **kwargs)

