import yaml


def load_yaml(filename):
    with open(filename, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise IOError(f"Could not load yaml: {exc}")


def save_yaml(filename, data):
    with open(filename, 'w') as file:
        return yaml.dump(data, file)
