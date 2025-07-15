import os
from coreComponents.partitioning import PartitioningModule
from coreComponents.trajectory_plugin import TrajectoryPlugin
from coreComponents.spatial_constraints_plugin import SpatialConstraintsPlugin
from coreComponents.tokenization import Tokenization
from coreComponents.detokenization import *

def use_operation(path:str,operation_name: str, partitioning_input_path: str, trajplug_input: dict):
    """
    Executes an operation on input trajectories.

    Args:
        operation_name (str): Name of the registered operation.
        partitioning_input_path (str): Path to the trajectories input file.
        trajplug_input (dict): Additional parameters for the operation logic.

    Returns:
        list: List of results per trajectory.
    """
    #Call the partitioning Module 
    
    PT = PartitioningModule(project_path=path)
    SC = SpatialConstraintsPlugin()
    TP = TrajectoryPlugin()
    operation_models,trajectories = PT.select_model(operation_name,partitioning_input_path)  
   
    # If multiple models, you could return all or pick the best one
    first_model_name = next(iter(operation_models))
    model_path = operation_models[first_model_name]
    print("Available Models in this region for this operation:",operation_models)
    print("Selected model: ",model_path)
    #TODO: Need to call a model instance using the model_path using the Transformers Plugin
    #so that I can pass this model instance to the logic function directly to be used.
    # Load operation logic and spatial constraints
    logic = TP.load(operation_name)
   
    # Load spatial rules (if needed)
    rules = SC.load(operation_name)  # implement if required
    print("Loaded rules for this operation: ",rules)
    # rules = None  # or empty list if unused for now

    # Tokenizer
    tokenizer = Tokenization()  

    final_results = []
    for traj in trajectories:
        tokenized_traj = tokenizer.tokenize(traj, resolution=9)  # resolution can be passed if needed
        print(tokenized_traj)
        result = logic(model_path, tokenized_traj, rules, **trajplug_input)
        # Results will be none for now as there is no called model/transformer in the logic.py for the operation of interest.
        print(result)
        # If result is spatial, decode tokens to points
        if tokenizer.is_spatial(result): #TODO: is is the best way to implement this function?
            continue
            # decoded_result = Detokenization.decode(result)
            # final_results.append(decoded_result)
        else:
            final_results.append(result)

    return final_results

    
