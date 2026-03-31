import yaml

def read_config(path: str) -> dict:
    '''Read in parameters from a config file'''

    # Open and read config file
    with open(path, 'r') as l:
        config = yaml.safe_load(l)

    return config