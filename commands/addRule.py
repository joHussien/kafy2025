from coreComponents.spatial_constraints_plugin import SpatialConstraintsPlugin

def AddOperation(rule_name, rule_script):
    if SpatialConstraintsPlugin.exists(rule_name):
        print(f"Rule '{rule_name}' already registered.")
        return None

    logic = SpatialConstraintsPlugin.load_functions_from_file(rule_script)
    if not hasattr(logic, 'apply_constraint'):
        raise AttributeError("Rule script must contain a function named 'apply_constraint'")
    
    SpatialConstraintsPlugin.add_rule(rule_name, rule_script)