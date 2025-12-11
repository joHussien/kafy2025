import os
import pickle
from coreComponents.partitioning import PartitioningModule
from coreComponents.operation_manager import OperationManager
from coreComponents.trajectory_plugin import TrajectoryPlugin
from coreComponents.spatial_constraints_plugin import SpatialConstraintsPlugin
from coreComponents.tokenization import Tokenization
from coreComponents.detokenization import *

def run_operation(path:str, operation_name: str, partitioning_input_path: str, trajplug_input: dict):
    """
    Executes an operation on input trajectories.

    Args:
        operation_name (str): Name of the registered operation.
        partitioning_input_path (str): Path to the trajectories input file.
        trajplug_input (dict): Additional parameters for the operation logic.

    Returns:
        list: List of results per trajectory.
    """
    # Call the partitioning Module 
    PT = PartitioningModule(project_path=path)
    SC = SpatialConstraintsPlugin()
    TP = TrajectoryPlugin()
    OM = OperationManager()
    
    # Get model path and trajectories
    model_path, trajectories = PT.select_model(operation_name, partitioning_input_path)  
    
    print(f"Selected model: {model_path}")
    
    # Load the model
    model_data = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Check if there's separate metadata file
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path).replace('.pkl', '').replace('.pt', '')
    metadata_path = os.path.join(model_dir, f"{model_name}_metadata.pkl")
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        tokenizer = metadata.get("tokenizer")
        transformer_name = metadata.get("transformer")
    else:
        # Fallback to model data
        tokenizer = model_data.get("tokenizer")
        transformer_name = model_data.get("transformer")
    
    # Load operation logic and spatial constraints
    logic_path = OM.get_path(operation_name)
    logic = SC.load_functions_from_file(logic_path)
    
    # Load spatial rules (if needed)
    rules = OM.get_rules(operation_name)
    rules_logic = SC.load_rules(rules) if rules else None
    print(f"Loaded rules for this operation: {rules}")
    
    # Tokenizer (use the one from the model if available)
    if tokenizer is None:
        tokenizer = Tokenization()
    
    final_results = []
    for traj in trajectories:
        tokenized_traj = tokenizer.tokenize(traj, resolution=9)
        print(f"Tokenized trajectory: {tokenized_traj}")
        
        # Pass the loaded model to the logic function
        result = logic(model=model_data, tokenized_traj=tokenized_traj, 
                      rules_logic=rules_logic, **trajplug_input)
        print(f"Result: {result}")
        
        # If result is spatial, decode tokens to points
        if tokenizer.is_spatial(result):  # TODO: is is the best way to implement this function?
            continue
            # decoded_result = Detokenization.decode(result)
            # final_results.append(decoded_result)
        else:
            final_results.append(result)

    return final_results