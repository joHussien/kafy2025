import importlib.util
from operation_manager import OperationManager
import os

class TrajectoryPlugin:
    @staticmethod
    def load(operation_name: str):
        """
        Dynamically loads the run_operation function from the operation's logic.py file.
        """
        logic_path = OperationManager.get_path(operation_name)

        if not os.path.exists(logic_path):
            raise FileNotFoundError(f"No logic file found for operation '{operation_name}'")

        spec = importlib.util.spec_from_file_location("logic_module", logic_path)
        logic_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(logic_module)

        if not hasattr(logic_module, "run_operation"):
            raise AttributeError(f"'run_operation' not defined in {logic_path}")

        return logic_module.run_operation