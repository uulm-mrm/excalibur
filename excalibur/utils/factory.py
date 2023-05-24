class FactoryError(Exception):
    def __init__(self, message=""):
        super().__init__(message)


def has_non_abstract_method(cls, name):
    return callable(getattr(cls, name, None)) and \
           (not hasattr(cls, '__abstractmethods__') or name not in cls.__abstractmethods__)


def get_child(base_cls, name, **kwargs):
    if base_cls.__name__ == name or (has_non_abstract_method(base_cls, 'name') and base_cls.name() == name):
        return base_cls(**kwargs)
    for sub_cls in base_cls.__subclasses__():
        try:
            return get_child(sub_cls, name, **kwargs)
        except FactoryError:
            pass
    raise FactoryError


def get_class_list(base_cls, include_base=False):
    if include_base:
        if has_non_abstract_method(base_cls, 'name'):
            classes = [[base_cls.name()]]
        else:
            classes = [[base_cls.__name__]]
    else:
        classes = [[]]

    for sub_cls in base_cls.__subclasses__():
        classes.append(get_class_list(sub_cls, include_base=True))
    classes = [item for sublist in classes for item in sublist]

    return classes
