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
def train_new_model(base_path,
                 operation_name,
                 training_data,
                 args):

    """
    Train an existing operation on new dataset:
        - validate operation exists
        - retireves all transformers registered for this operation
        - tokenize dataset (operation-dependent)
        - train model
        - save model using partitioning module
        - ONLY THEN update operation registry
    """

    PT = PartitioningModule(base_path)

     # ----------------------------------------------------------
    # 1. Load training CSV
    # ----------------------------------------------------------
    df = pd.read_csv(training_data)

    # ----------------------------------------------------------
    # 2. Tokenization (operation dependent)
    # ----------------------------------------------------------
    tokenizer = TrajectoryTokenizer(resolution=8)

    tokenized = tokenize_dataset(
        operation_type=args["operation"],
        df=df,
        tokenizer=tokenizer,
    )
    logging.info(f"Training data got tokenized successfully.")
    # ----------------------------------------------------------
    # 3. Retireve all transofmers for this operation
    # ----------------------------------------------------------
    tplugin = TransformersPlugin(base_path)
    oplugin = TrajectoryOperationsPlugin(base_path)
    operation_data = oplugin.get(operation_name)
    registered_transformers = operation_data['transformers']
    logging.info(f"For this operation, found Transformer(s) {registered_transformers}")
    for transformer in registered_transformers:
        build_model_fn = tplugin.load_model_builder(transformer)

        if build_model_fn is None:
            logger.error(f"Transformer '{transformer}' not found.")
            raise ValueError(f"Transformer '{transformer}' not found.")
        # ----------------------------------------------------------
        # 4. Make sure the operation already has this transformer (should be yes)
        # ----------------------------------------------------------
        
        if not oplugin.has_transformer(operation_name, transformer):
            logger.error(
                f"Operation '{operation_name}' does not have this transformer "
                f"'{transformer}' registered."
            )
            logger.error(
                "Training aborted. Please add this transformer first to this operation."
            )
            return
        # ----------------------------------------------------------
        # 6. Train this transformer model (common training logic)
        # ----------------------------------------------------------
        model = train_operation_model(
            build_model_fn,
            tokenizer,
            tokenized,
            args,
        )
        logging.info(f"Transformer {transformer} was trained successfully for Operation {operation_name}")
        # print(args)
        # ----------------------------------------------------------
        # 7. Save model using Partitioning Module
        # ----------------------------------------------------------
        # Initialize partitioning module
        # partitioning = PartitioningModule(base_path)
        
        # Create a unique model name based on operation and transformer
        model_name = f"{operation_name}_{transformer}"
        
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
            "transformer": transformer,
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

        logging.info(f"Training new model for Operation '{operation_name}' completed successfully.")
