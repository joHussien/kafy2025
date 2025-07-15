# core/model_executor.py

from typing import Any, Dict, List
from shapely.geometry import Point
import importlib.util
import os
from core.partitioning import PartitioningModule

class ModelExecutor:
    def __init__(self, project_path: str):
        self.partitioning_module = PartitioningModule(project_path)
        self.project_path = project_path

    def execute_operation(
        self,
        operation_name: str,
        trajectory: List[Point],
        traj_input: Dict[str, Any]
    ) -> Any:
        # 1. Find the model path via partitioning
        enclosing_cell = self.partitioning_module._find_enclosing_cell_of_trajectory_list([trajectory])
        if not enclosing_cell or not enclosing_cell["occupied"]:
            raise ValueError(f"No model found for trajectory in operation '{operation_name}'.")

        model_path = enclosing_cell["model_path"]
        model_metadata = enclosing_cell["metadata"]

        # 2. Load the operation plug-in (architecture.py)
        plugin_path = os.path.join(self.project_path, "operations", operation_name, "architecture.py")
        spec = importlib.util.spec_from_file_location("architecture", plugin_path)
        architecture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(architecture)

        # 3. Initialize the model (assumes load_model() is defined in architecture)
        model = architecture.load_model(model_path)

        # 4. Run inference using the model (assumes inference() is defined in architecture)
        output = architecture.inference(model, trajectory, traj_input)

        return output