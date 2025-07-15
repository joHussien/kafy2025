REGISTRY = {}

def add_operation(name, model, logic_fn, rules):
    REGISTRY[name] = {
        'model': model,
        'logic': logic_fn,
        'rules': rules
    }

def get_operation(name):
    return REGISTRY.get(name)