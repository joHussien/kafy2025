# This code is WRONG but keep it for now.

# addOperation.py

import os
from coreComponents.transformers_plugin import TransformersPlugin
from coreComponents.trajectory_plugin import TrajectoryPlugin
from coreComponents.spatial_constraints_plugin import SpatialConstraintsPlugin
#from core.registry import Registry --> ??

def AddOperation(operation_name, transformer, operation_script, spatial_constraints=None):
    if Registry.exists(operation_name):
        return f"Operation '{operation_name}' already registered."

    # Load and register transformer model
    model = TransformersPlugin.load(transformer)
    TransformersPlugin.register(operation_name, model)

    # Load and register operation logic
    logic = SpatialConstraintsPlugin.load_function_from_file(operation_script)
    if not hasattr(logic, 'operation_logic'):
        raise AttributeError("Operation script must contain a function named 'operation_logic'")
    logic_fn = logic.operation_logic
    TrajectoryPlugin.register(operation_name, logic_fn)

    rules = []
    if spatial_constraints is not None and os.path.exists(spatial_constraints):
        rules = SpatialConstraintsPlugin.preprocess_constraints_file(
            constraints_file_path=spatial_constraints,
            operation_name=operation_name
        )

    Registry.add(operation_name, model, logic_fn, rules)
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
