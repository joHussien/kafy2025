# This code is WRONG but keep it for now.

# addOperation.py

import os
import json
from coreComponents.transformers_plugin import TransformersPlugin
# from coreComponents.trajectory_plugin import TrajectoryPlugin
from coreComponents.spatial_constraints_plugin import SpatialConstraintsPlugin
from coreComponents.operation_manager import OperationManager

def AddOperation(operation_name, transformer, operation_script, spatial_constraints=None):
    if OperationManager.exists(operation_name):
        return f"Operation '{operation_name}' already registered."

    # Load and register transformer model
    if not TransformersPlugin.exists(transformer):
        raise AttributeError("Transformer is not registered in the transformers plugin. Add Transformer first!")

    # Load and register operation logic
    logic = SpatialConstraintsPlugin.load_function_from_file(operation_script)
    if not hasattr(logic, 'operation_logic'):
        raise AttributeError("Operation script must contain a function named 'operation_logic'")

    rules = []
    if spatial_constraints is not None and os.path.exists(spatial_constraints):
        # TODO: Figure out the format. For now, assume it's a json list
        with open(spatial_constraints, "r") as constraints_file:
            constraints = json.load(constraints_file)
        
        for constraint in constraints:
            if SpatialConstraintsPlugin.exists(constraint):
                rules.append(constraint)
            else:
                raise AttributeError("Constraint \"" + constraint + "\" does not exist in constraints registry")


    OperationManager.add_operation(operation_name, transformer, operation_script, rules)
    return f"Operation '{operation_name}' successfully added."

# Optional CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add a new operation to the system.")
    parser.add_argument("OperationName")
    parser.add_argument("--Transformer", required=True)
    parser.add_argument("--OperationScript", required=True)
    parser.add_argument("--SpatialConstraints", required=False)

    args = parser.parse_args()

    result = AddOperation(
        operation_name=args.OperationName,
        transformer=args.Transformer,
        operation_script=args.OperationScript,
        spatial_constraints=args.SpatialConstraints
    )
    print(result)
