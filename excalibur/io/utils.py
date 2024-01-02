import json

import yaml


def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        return data


def save_json(filename, data, sort_keys=False):
    json_data = json.dumps(data, sort_keys=sort_keys)
    with open(filename, 'w') as f:
        f.write(json_data)


def load_yaml(filename):
    with open(filename, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise IOError(f"Could not load yaml: {exc}")


def save_yaml(filename, data, sort_keys=False):
    with open(filename, 'w') as file:
        return yaml.dump(data, file, sort_keys=sort_keys)
