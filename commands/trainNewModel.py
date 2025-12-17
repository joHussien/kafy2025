# commands/addOperation.py

import os
import pandas as pd
import torch
import pickle

from coreComponents.transformers_plugin import TransformersPlugin
from coreComponents.tokenization import TrajectoryTokenizer
from coreComponents.dataCollator import TrajectoryDataCollator
from coreComponents.trajectory_operations_plugin import TrajectoryOperationsPlugin
from coreComponents.offline_training import train_operation_model
from coreComponents.partitioning import PartitioningModule
from commands.addOperation import *
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

    #  # ----------------------------------------------------------
    # # 1. Load training CSV
    # # ----------------------------------------------------------
    # df = pd.read_csv(training_data)

    # # ----------------------------------------------------------
    # # 2. Tokenization (operation dependent)
    # # ----------------------------------------------------------
    # tokenizer = TrajectoryTokenizer(resolution=8)

    # tokenized = tokenize_dataset(
    #     operation_type=args["operation"],
    #     df=df,
    #     tokenizer=tokenizer,
    # )
    # logging.info(f"Training data got tokenized successfully.")
    # ----------------------------------------------------------
    # 3. Retireve all transofmers for this operation
    # ----------------------------------------------------------
    tplugin = TransformersPlugin(base_path)
    oplugin = TrajectoryOperationsPlugin(base_path)
    operation_data = oplugin.get(operation_name)
    registered_transformers = operation_data['transformers']
    logging.info(f"For this operation, found Transformer(s) {registered_transformers}")

    # ----------------------------------------------------------
    # 4. Load training CSV
    # ----------------------------------------------------------
    df = pd.read_csv(training_data)
    logger.info(f"Loaded training data with {len(df)} rows")

    # ----------------------------------------------------------
    # 5. NEW: Process CSV based on operation type
    # ----------------------------------------------------------
    tokenizer = TrajectoryTokenizer(resolution=args.get("resolution", 10))
    
    # Determine which columns to tokenize based on operation type
    operation_type = args["operation"]
    
    if operation_type == "summarization":
        columns_to_tokenize = ["trajectory", "summary"]
    elif operation_type == "generation":
        columns_to_tokenize = ["trajectory"]
    elif operation_type == "classification":
        columns_to_tokenize = ["trajectory"]  # labels are already numeric
    elif operation_type == "next_point":
        columns_to_tokenize = ["trajectory"]
    else:
        raise ValueError(f"Unknown operation type: {operation_type}")
    
    # Process CSV: Convert GPS strings to H3 tokens
    logger.info(f"Converting GPS to H3 tokens for columns: {columns_to_tokenize}")
    df_processed = TrajectoryTokenizer.process_csv(
        df=df,
        columns_to_tokenize=columns_to_tokenize,
        resolution=tokenizer.resolution
    )
    print(df_processed)
    # Build vocabulary from processed data
    logger.info("Building vocabulary...")
    all_h3_strings = []
    for col in columns_to_tokenize:
        if col in df_processed.columns:
            col_h3_strings = df_processed[col].dropna().tolist()
            all_h3_strings.extend(col_h3_strings)
    
    tokenizer.build_vocab(all_h3_strings)
    logger.info(f"Vocabulary built with {tokenizer.get_vocab_size()} tokens")
    
    # ----------------------------------------------------------
    # 6. NEW: Prepare dataset for the specific operation
    # ----------------------------------------------------------
    logger.info(f"Preparing dataset for {operation_type} operation...")
    
    if operation_type == "summarization":
        dataset = prepare_summarization_dataset(df_processed, tokenizer)
    elif operation_type == "generation":
        dataset = prepare_generation_dataset(df_processed, tokenizer)
    elif operation_type == "classification":
        dataset = prepare_classification_dataset(df_processed, tokenizer)
    elif operation_type == "next_point":
        dataset = prepare_next_point_dataset(df_processed, tokenizer)
    else:
        raise ValueError(f"Unsupported operation type: {operation_type}")
    # ----------------------------------------------------------
    # 6.5. Save processed H3 dataset to CSV
    # ----------------------------------------------------------
    
    saved_tokenized_data_path = PT.save_tokenized_dataset(tokenized_dataset=df_processed,
        operation_name=operation_name,
        original_data_path=training_data)
    logging.info(f"Tokenized Data saved to: {saved_tokenized_data_path}")
    # ----------------------------------------------------------
    # 7. Create data collator based on operation type
    # ----------------------------------------------------------
    collator = create_data_collator(tokenizer, operation_type, args.get("max_length", 256))
    

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
        logging.info(f"Started training Transformer {transformer} for Operation {operation_name}")
        model = train_operation_model(
            build_model_fn,
            tokenizer,
            dataset,
            collator,
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
