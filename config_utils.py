import json

config = None

def load_config():
    global config	
    with open('config.json') as f:
        config = json.loads(f.read())

def get(param, default_value=None):
    if config == None:
        load_config()
    return config.get(param, default_value)
