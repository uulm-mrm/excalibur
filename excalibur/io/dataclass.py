from dataclasses import asdict, is_dataclass

import motion3d as m3d
import numpy as np

from excalibur.io.utils import load_yaml, save_yaml, load_json, save_json


class DataclassIO:
    @classmethod
    def _value_from_dict(cls, k, v, field_type=None):
        if k is not None:
            field = cls.__dataclass_fields__[k]
            field_type = field.type

        if hasattr(field_type, '__dict__'):
            if '_name' in field_type.__dict__:
                if field_type.__dict__['_name'] == 'Dict':
                    value_field_type = field_type.__dict__['__args__'][1]
                    return {a: cls._value_from_dict(None, b, field_type=value_field_type) for a, b in v.items()}
                if field_type.__dict__['_name'] == 'List':
                    value_field_type = field_type.__dict__['__args__'][0]
                    return [cls._value_from_dict(None, a, field_type=value_field_type) for a in v]
                if field_type.__dict__['_name'] == 'Optional':
                    if v is None:
                        return None
                    else:
                        value_field_type = field_type.__dict__['__args__'][0]
                        return cls._value_from_dict(None, v, field_type=value_field_type)

        if field_type == m3d.TransformInterface:
            # TODO(horn): support other transforms
            assert len(v) == 6
            return m3d.EulerTransform(v[:3], np.deg2rad(v[3:]))
        if field_type == np.ndarray:
            return np.array(v)

        if is_dataclass(field_type):
            return field_type.from_dict(v)

        return v

    @classmethod
    def from_dict(cls, data):
        data = {k: cls._value_from_dict(k, v) for k, v in data.items()}
        return cls(**data)

    @classmethod
    def _obj_to_dict(cls, obj):
        if is_dataclass(obj):
            return {k: cls._obj_to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif np.issubdtype(type(obj), np.integer):
            return int(obj)
        elif np.issubdtype(type(obj), np.floating):
            return float(obj)
        else:
            # TODO(horn): support m3d.TransformInterface
            return obj

    def to_dict(self):
        return self._obj_to_dict(self)

    @classmethod
    def from_yaml(cls, filename):
        data = load_yaml(filename)
        return cls.from_dict(data)

    def to_yaml(self, filename, sort_keys=False):
        data = self.to_dict()
        save_yaml(filename, data, sort_keys=sort_keys)

    @classmethod
    def from_json(cls, filename):
        data = load_json(filename)
        return cls.from_dict(data)

    def to_json(self, filename):
        data = self.to_dict()
        save_json(filename, data)
