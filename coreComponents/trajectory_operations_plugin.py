# coreComponents/trajectory_operations_plugin.py

import os
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REG_FILE = "operations.json"


class TrajectoryOperationsPlugin:

    def __init__(self, base_path):
        self.base = base_path
        self.path = os.path.join(base_path, REG_FILE)

        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({}, f, indent=4)

    # ----------------------------------------------------------
    # Load registry
    # ----------------------------------------------------------
    def load(self):
        with open(self.path, "r") as f:
            return json.load(f)

    # ----------------------------------------------------------
    # Function 1: Check if operation already has this transformer
    # ----------------------------------------------------------
    def has_transformer(self, op_name, transformer_name):
        """Check if operation already has this transformer registered."""
        registry = self.load()
        if op_name not in registry:
            return False
        existing_transformers = registry[op_name].get("transformers", [])
        return transformer_name in existing_transformers

    # ----------------------------------------------------------
    # Function 2: Register operation AFTER successful training
    # ----------------------------------------------------------
    def register(self, op_name, script_path, transformer_name, model_path):
        """
        Register an operation with a transformer.
        Call this ONLY AFTER training is successful.
        """
        registry = self.load()

        if op_name not in registry:
            # Create new operation entry
            registry[op_name] = {
                "transformers": [transformer_name],
                "script": script_path
            }
            logging.info(f"New operation '{op_name}' registered with transformer '{transformer_name}'")
        else:
            # Operation already exists, add transformer to the list
            existing_transformers = registry[op_name].get("transformers", [])
            
            if transformer_name not in existing_transformers:
                existing_transformers.append(transformer_name)
                registry[op_name]["transformers"] = existing_transformers
                logging.info(f"Added transformer '{transformer_name}' to existing operation '{op_name}'")
            

        with open(self.path, "w") as f:
            json.dump(registry, f, indent=4)

    # ----------------------------------------------------------
    # Lookup operation
    # ----------------------------------------------------------
    def get(self, op_name):
        registry = self.load()
        return registry.get(op_name, None)

    def exists(self, op_name):
        registry = self.load()
        return op_name in registry