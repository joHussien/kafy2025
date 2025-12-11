# commands/addOperation.py

import os
import pandas as pd
import torch
import pickle

from coreComponents.transformers_plugin import TransformersPlugin
from coreComponents.tokenization import (
    TrajectoryTokenizer,
    tokenize_dataset,
)
from coreComponents.trajectory_operations_plugin import TrajectoryOperationsPlugin
from coreComponents.offline_training import train_operation_model
from coreComponents.partitioning import PartitioningModule
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
def add_operation(base_path,
                 operation_name,
                 transformer_name,
                 training_data,
                 args,
                 operation_script):

    """
    Add a new operation:
        - validate transformer exists
        - check if operation already has this transformer
        - tokenize dataset (operation-dependent)
        - train model
        - save model using partitioning module
        - ONLY THEN update operation registry
    """

    # ----------------------------------------------------------
    # 1. Validate transformer exists
    # ----------------------------------------------------------
    PT = PartitioningModule(base_path)
    tplugin = TransformersPlugin(base_path)
    build_model_fn = tplugin.load_model_builder(transformer_name)

    if build_model_fn is None:
        logger.error(f"Transformer '{transformer_name}' not found.")
        raise ValueError(f"Transformer '{transformer_name}' not found.")

    # ----------------------------------------------------------
    # 2. Check if operation already has this transformer
    # ----------------------------------------------------------
    oplugin = TrajectoryOperationsPlugin(base_path)
    
    # Function 1: Check if transformer already exists for this operation
    if oplugin.has_transformer(operation_name, transformer_name):
        logger.error(
            f"Operation '{operation_name}' already has transformer "
            f"'{transformer_name}' registered."
        )
        logger.error(
            "Training aborted. Use a different transformer name or operation name."
        )
        return

    # ----------------------------------------------------------
    # 3. Validate operation script exists
    # ----------------------------------------------------------
    script_path = os.path.abspath(operation_script)
    if not os.path.exists(script_path):
        logger.error(f"Operation script not found at path: {script_path}")
        raise FileNotFoundError(f"Operation script not found: {script_path}")

    # ----------------------------------------------------------
    # 4. Load training CSV
    # ----------------------------------------------------------
    df = pd.read_csv(training_data)

    # ----------------------------------------------------------
    # 5. Tokenization (operation dependent)
    # ----------------------------------------------------------
    tokenizer = TrajectoryTokenizer(resolution=8)

    tokenized = tokenize_dataset(
        operation_type=args["operation"],
        df=df,
        tokenizer=tokenizer,
    )
    logging.info(f"Training data got tokenized successfully.")
    # ----------------------------------------------------------
    # 6. Train model (common training logic)
    # ----------------------------------------------------------
    model = train_operation_model(
        build_model_fn,
        tokenizer,
        tokenized,
        args,
    )
    logging.info(f"New model trained successfully")
    # print(args)
    # ----------------------------------------------------------
    # 7. Save model using Partitioning Module
    # ----------------------------------------------------------
    # Initialize partitioning module
    # partitioning = PartitioningModule(base_path)
    
    # Create a unique model name based on operation and transformer
    model_name = f"{operation_name}_{transformer_name}"
    
    # Save the model using partitioning module
    model_path = PT.save_model(
        model=model,
        operation_name=operation_name,
        model_name=model_name,
        training_trajectories_path=training_data  # Path to CSV file
    )
    
    # ----------------------------------------------------------
    # 8. Save tokenizer and transformer info with the model
    # ----------------------------------------------------------
    model_info = {
        "state_dict": model.state_dict(),
        "tokenizer": tokenizer,
        "transformer": transformer_name,
        "operation_name": operation_name,
        "model_name": model_name,
        "training_data_path": training_data,
    }
    
    # Save additional metadata alongside the model
    metadata_path = os.path.join(os.path.dirname(model_path), f"{model_name}_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(model_info, f)

    # ----------------------------------------------------------
    # 9. Function 2: ONLY NOW update the operations.json registry
    # ----------------------------------------------------------
    oplugin.register(operation_name, script_path, transformer_name, model_path)

    logging.info(f"Adding/Updating Operation '{operation_name}' completed successfully.")
