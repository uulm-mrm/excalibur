import argparse


class ParseEnum(argparse.Action):
    def __init__(self, option_strings, enum_type, *args, **kwargs):
        self._enum_type = enum_type
        kwargs['choices'] = [f.name for f in list(enum_type)]
        if 'default' not in kwargs:
            kwargs['default'] = None
        super(ParseEnum, self).__init__(option_strings=option_strings, *args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, (list, tuple)):
            value = str(values[0])
        else:
            value = str(values)
        try:
            enum_value = self._enum_type[value]
            setattr(namespace, self.dest, enum_value)
        except KeyError:
            parser.error('Input {} is not a field of enum {}'.format(values, self._enum_type))
