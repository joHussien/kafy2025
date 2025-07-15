import os
from Partitioning import PartitioningModule
from TransformersPlugin import TransformersPlugin
from Tokenization import Tokenization
from Training import start_training  # Assume you have a training function here
from Registry import Registry

def train_new_model(operation_name: str, dataset_path: str, training_args: dict):
    """
    Trains a new model for the specified operation using the given dataset and training arguments.
    """
    # Load dataset trajectories (assumed JSON or similar file format)
    dataset = load_dataset(dataset_path)  # You need to implement this loading function
    
    # Extract trajectories from dataset
    trajectories = dataset["trajectories"]  # Adjust key accordingly
    
    # Initialize partitioning module
    partitioning = PartitioningModule(project_path="path_to_your_project")  # adjust project path
    
    # Compute MBR of trajectories
    location = partitioning.calculate_mbr_gps(trajectories)
    
    # Get model instance from TransformerPlugin for this operation
    model = TransformersPlugin.get_model(operation_name)
    if model is None:
        raise ValueError(f"No model registered for operation {operation_name}")
    
    # Tokenize dataset for training
    tokenized_data = Tokenization.tokenize_train_data(dataset, operation_name)
    
    # Start training
    start_training(model, tokenized_data, training_args)
    
    # Save trained model with partitioning
    partitioning.save_model(model, location, operation_name)
    
    # Register the new model in the Registry
    Registry.add(operation_name, model)
    
    print(f"Model trained and saved at location covering {location}")
    return location

# --- Helper: Implement dataset loader ---
def load_dataset(path):
    import json
    with open(path, 'r') as f:
        return json.load(f)