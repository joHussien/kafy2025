# coreComponents/offline_training.py

import torch
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
class CustomTrainingArguments(TrainingArguments):
    """
    Custom TrainingArguments that allows overriding defaults with args dictionary.
    Automatically filters out non-TrainingArguments parameters.
    """
    
    def __init__(self, args_dict: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize with args dictionary to override defaults.
        
        Args:
            args_dict: Dictionary of arguments (will filter out non-training args)
            **kwargs: Additional keyword arguments (will override args_dict)
        """
        # Get all valid TrainingArguments parameters from parent class
        import inspect
        parent_init = inspect.signature(TrainingArguments.__init__)
        valid_params = list(parent_init.parameters.keys())
        
        # Define our default values
        default_args = {
            "output_dir": "./tmp",
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "num_train_epochs": 3,
            "learning_rate": 5e-4,
            "evaluation_strategy": "steps",
            "eval_steps": 100,
            "save_steps": 200,
            "logging_steps": 50,
            "report_to": "none",
            "disable_tqdm": False,
            "remove_unused_columns": False,
        }
        
        # Start with default_args
        final_args = default_args.copy()
        
        # Override with args_dict if provided (only valid parameters)
        if args_dict:
            for key, value in args_dict.items():
                # Skip non-training args like 'operation'
                if key in valid_params:
                    final_args[key] = value
                # Handle common custom mappings
                elif key == "batch_size":
                    final_args["per_device_train_batch_size"] = value
                    final_args["per_device_eval_batch_size"] = value
                elif key == "epochs":
                    final_args["num_train_epochs"] = value
                elif key == "lr":
                    final_args["learning_rate"] = value
                # Silently ignore other non-training args
        
        # Override with explicit kwargs (highest priority)
        final_args.update(kwargs)
        
        # Initialize the parent class with all arguments
        super().__init__(**final_args)
def collate_fn(batch):
    """
    Very simple padding collator.
    Works because our tokenizer uses integer ids.
    """
    max_len = max(len(x["input_ids"]) for x in batch)

    def pad(seq):
        return seq + [0] * (max_len - len(seq))

    input_ids = [pad(x["input_ids"]) for x in batch]
    labels = [pad(x["labels"]) for x in batch]

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
    }


def train_operation_model(build_model_fn,
                          tokenizer,
                          tokenized_dataset,
                          collator,
                          args):
    """ 
    Common training pipeline for ALL operations.

    build_model_fn: HF-style class, e.g. T5ForConditionalGeneration
    tokenizer:       TrajectoryTokenizer instance
    tokenized_dataset: DatasetDict (train + validation)
    args:            dict containing training hyperparams

    Returns:
        Trained model instance
    """

    # ----------------------------------------------------------
    # 1. Instantiate model
    # ----------------------------------------------------------
    # if build_model_fn.__name__ == "T5":
    model = build_model_fn(tokenizer)
    # print(model)
        # model = build_model_fn(config)

    # else:
    #     # fallback: do a generic scratch config
    #     from transformers import AutoConfig
    #     cfg = AutoConfig.from_model_type(build_model_fn.__name__)
    #     model = build_model_fn(cfg)


    # ----------------------------------------------------------
    # 2. Training arguments (HF style)
    # ----------------------------------------------------------
    training_args = CustomTrainingArguments(args_dict=args)

    # ----------------------------------------------------------
    # 3. Trainer
    # ----------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=collator,
    )

    # ----------------------------------------------------------
    # 4. Train
    # ----------------------------------------------------------
    trainer.train()

    return model