"""
Partitioning.py
"""

import os
import json
import logging
from math import sqrt
from typing import List, Tuple
from shapely.geometry import Point
import pandas as pd
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class UnexpectedError(Exception):
    def __init__(self, message="An unexpected error occurred. This should not happen."):
        super().__init__(message)

class PartitioningModule:
    def __init__(self, project_path):
        self.models_repo_path = os.path.join(project_path, "modelsRepository")
        self.config_file = os.path.join(project_path,  "modelsRepository","pyramidConfigs.json")
        self.pyramid_path = os.path.join(project_path,  "modelsRepository","partitioningPyramid.json")
        self.pyramid = {}
        self.tokens_threshold_per_cell = 100
        os.makedirs(self.models_repo_path, exist_ok=True)

        self.load_config()

        if self.build_pyramid_flag:
            self.build_pyramid(self.pyramid_path)
            with open(self.config_file, 'r+') as file:
                data = json.load(file)
                data["build_pyramid_from_scratch"] = False
                file.seek(0)
                json.dump(data, file, indent=4)
                file.truncate()
        else:
            self.pyramid = self.load_pyramid(self.pyramid_path)

    def select_model(self, operation_name: str, partitioning_input_path: str) -> Tuple[str, List[List[Point]]]:
        """
        Select the most appropriate model for a given operation and input trajectories.

        Args:
            operation_name (str): The name of the operation (e.g., 'classification').
            partitioning_input_path (str): Path to the input file containing the trajectories.

        Returns:
            model_path (str): Path to the selected model.
            trajectories (List[List[Point]]): Parsed trajectories.
        """
        # 1. Load trajectories from input file
        trajectories = self.load_trajectories(partitioning_input_path)
        if not trajectories:
            raise ValueError("No trajectories loaded from the input path.")

        # 2. Find the enclosing cell for this trajectory batch
        cell = self._find_enclosing_cell_of_trajectory_list(trajectories)
        if not cell:
            raise ValueError("No enclosing cell found for the given trajectory(s).")

        # 3. Check if the operation exists in this cell
        if "models" not in cell or operation_name not in cell["models"]:
            raise ValueError(f"No model found for operation '{operation_name}' in this region.")

        # 4. Get all models for this operation (returns a dict of model_name → path)
        operation_models = cell["models"][operation_name]
        
        # For now, return the first model found (later we'll implement optimization)
        if not operation_models:
            raise ValueError(f"No models available for operation '{operation_name}' in this cell.")
        
        # Return the first model path
        first_model_name = list(operation_models.keys())[0]
        model_path = operation_models[first_model_name]
        
        return model_path, trajectories
    @staticmethod
    def load_trajectories(path: str) -> List[List[Point]]:
        """
        Loads trajectories from a CSV file.
        
        Args:
            path (str): Path to the CSV file.

        Returns:
            List[List[Point]]: A list of trajectories, where each trajectory is a list of shapely Point objects.
        """
        try:
            # Read CSV file
            df = pd.read_csv(path)
            
            trajectories = []
            
            # Check which column contains trajectory data
            # Priority: 'trajectory' column, then any column containing 'trajectory' in name
            trajectory_column = None
            for col in df.columns:
                if col.lower() == 'trajectory':
                    trajectory_column = col
                    break
                elif 'trajectory' in col.lower():
                    trajectory_column = col
                    # Don't break, keep looking for exact match
            
            if trajectory_column is None:
                # If no trajectory column found, check if only one column exists
                if len(df.columns) == 1:
                    trajectory_column = df.columns[0]
                else:
                    raise ValueError(f"No trajectory column found in CSV file. Columns: {df.columns.tolist()}")
            
            # Process each row
            for idx, row in df.iterrows():
                trajectory_str = str(row[trajectory_column]).strip()
                
                # Skip empty rows
                if not trajectory_str or trajectory_str.lower() == 'nan':
                    continue
                
                # Remove quotes if present
                trajectory_str = trajectory_str.strip('"\'')
                
                # Parse the trajectory string
                try:
                    # Split by commas to get individual GPS points
                    points_str = trajectory_str.split(',')
                    trajectory = []
                    
                    for point_str in points_str:
                        point_str = point_str.strip()
                        if not point_str:
                            continue
                        
                        # Split by space or comma to get lat and lon
                        # Handle both "lat lon" and "lat,lon" formats
                        if ' ' in point_str:
                            parts = point_str.split()
                        elif ',' in point_str:
                            parts = point_str.split(',')
                        else:
                            continue
                        
                        if len(parts) >= 2:
                            try:
                                # Try to parse as floats
                                lat = float(parts[0].strip())
                                lon = float(parts[1].strip())
                                point = Point(lon, lat)  # Note: Point takes (x, y) = (lon, lat)
                                trajectory.append(point)
                            except ValueError:
                                logging.warning(f"Could not parse GPS point: {point_str}")
                                continue
                    
                    if trajectory:  # Only add non-empty trajectories
                        trajectories.append(trajectory)
                        
                except Exception as e:
                    logging.warning(f"Error parsing trajectory at row {idx}: {e}")
                    continue
            
            if not trajectories:
                logging.warning(f"No valid trajectories found in {path}")
            
            logging.info(f"Loaded {len(trajectories)} trajectories from {path}")
            return trajectories
            
        except Exception as e:
            logging.error(f"Error loading trajectories from {path}: {e}")
            return []
    def load_config(self):
        default_configs = {"H": 5, "L": 3, "build_pyramid_from_scratch": True}
        if not os.path.isfile(self.config_file):
            with open(self.config_file, "w", encoding="utf-8") as file:
                json.dump(default_configs, file, indent=4)
            logging.warning("Pyramid Configurations File not found. Assigned default configs.")

        with open(self.config_file, "r", encoding="utf-8") as file:
            config = json.load(file)
            self.pyramid_height = config.get("H", 5)
            self.pyramid_levels = config.get("L", 3)
            self.build_pyramid_flag = config.get("build_pyramid_from_scratch",True)

    def _calculate_bounds(self, h, index):
        num_cells = 4**h
        cells_per_side = int(sqrt(num_cells))
        lat_step = 180 / cells_per_side
        lon_step = 360 / cells_per_side
        row = index // cells_per_side
        col = index % cells_per_side
        min_lat = round(90 - (row * lat_step), 6)
        max_lat = round(min_lat - lat_step, 6)
        min_lon = round(-180 + (col * lon_step), 6)
        max_lon = round(min_lon + lon_step, 6)
        return (min_lat, max_lat, min_lon, max_lon)

    def _generate_cells(self, h):
        num_cells = 4**h
        return {
            i: {
               "height": h,
                "index": i,
                "bounds": self._calculate_bounds(h, i),
                "occupied": False,
                "models": {},     # New structure for storing operation -> model_name -> path
                "metadata": {},   # New structure for storing operation -> list of model names
                "num_tokens": 0,
            } for i in range(num_cells)
        }

    def build_pyramid(self, location):
        self.pyramid = {l: self._generate_cells(l) for l in range(self.pyramid_height + 1)}
        os.makedirs(os.path.dirname(location), exist_ok=True)
        with open(location, "w", encoding="utf-8") as file:
            json.dump(self.pyramid, file, indent=4)
        logging.info("Successfully built the partitioning pyramid.")

    def load_pyramid(self, pyramid_path=None):
        pyramid_path = pyramid_path or self.pyramid_path
        if os.path.exists(pyramid_path):
            with open(pyramid_path, "r", encoding="utf-8") as file:
                return json.load(file)
        else:
            raise FileNotFoundError(f"Pyramid file not found at {pyramid_path}")

    def save_pyramid(self):
        os.makedirs(os.path.dirname(self.pyramid_path), exist_ok=True)
        with open(self.pyramid_path, "w", encoding="utf-8") as file:
            json.dump(self.pyramid, file, indent=4)

    def calculate_mbr_gps(self, trajectory_list: List[List[Point]]) -> Tuple[float, float, float, float]:
        """
        Calculates the Minimum Bounding Rectangle (MBR) for a set of trajectories.

        Args:
            trajectory_list (List[List[Point]]): List of trajectories, each a list of shapely Point objects.

        Returns:
            Tuple[float, float, float, float]: (min_lat, max_lat, min_lon, max_lon)
        """
        min_lat = min_lon = float("inf")
        max_lat = max_lon = float("-inf")

        for trajectory in trajectory_list:
            for point in trajectory:
                min_lat = min(min_lat, point.y)
                max_lat = max(max_lat, point.y)
                min_lon = min(min_lon, point.x)
                max_lon = max(max_lon, point.x)

        return round(min_lat, 6), round(max_lat, 6), round(min_lon, 6), round(max_lon, 6)

    def _is_bounding_rectangle_enclosed(self, rect, cell_bounds):
        lat_min, lat_max, lon_min, lon_max = rect
        cell_lat_max, cell_lat_min, cell_lon_min, cell_lon_max = cell_bounds
        return (
            lat_min >= cell_lat_min and
            lat_max <= cell_lat_max and
            lon_min >= cell_lon_min and
            lon_max <= cell_lon_max
        )

    def _find_enclosing_cell_of_trajectory_list(self, trajectory_list: List[List[Point]]):
        bounding_rectangle = self.calculate_mbr_gps(trajectory_list)
        for l in reversed(range(self.pyramid_height + 1)):
            for i, cell in self.pyramid[str(l)].items():
                if self._is_bounding_rectangle_enclosed(bounding_rectangle, cell["bounds"]):
                    return cell
        return None

    def _update_cell_with_model(self, operation_name: str, model_name: str, 
                          model_path: str, cell: dict, num_tokens: int):
        """
        Update cell metadata with a new model.
        
        Args:
            operation_name (str): Name of the operation
            model_name (str): Name/identifier of the model
            model_path (str): Path to the saved model
            cell (dict): The cell to update
            num_tokens (int): Number of trajectories used for training
        """
        # Get cell level and index
        cell_level = cell["height"]
        cell_index = cell["index"]
        
        # Get the actual cell from pyramid (converted to string keys for access)
        level_key = str(cell_level)
        index_key = str(cell_index)
        
        # Make sure the cell exists in pyramid
        if level_key not in self.pyramid or index_key not in self.pyramid[level_key]:
            raise KeyError(f"Cell {cell_level}_{cell_index} not found in pyramid")
        
        pyramid_cell = self.pyramid[level_key][index_key]
        
        # Initialize models dictionary if not exists
        if "models" not in pyramid_cell:
            pyramid_cell["models"] = {}
        
        # Initialize operation entry if not exists
        if operation_name not in pyramid_cell["models"]:
            pyramid_cell["models"][operation_name] = {}
        
        # Add model to operation
        pyramid_cell["models"][operation_name][model_name] = model_path
        
        # Update metadata
        if "metadata" not in pyramid_cell:
            pyramid_cell["metadata"] = {}
        
        if operation_name not in pyramid_cell["metadata"]:
            pyramid_cell["metadata"][operation_name] = []
        
        # Add model name to metadata if not already present
        if model_name not in pyramid_cell["metadata"][operation_name]:
            pyramid_cell["metadata"][operation_name].append(model_name)
        
        # Update cell statistics
        pyramid_cell["occupied"] = True
        pyramid_cell["num_tokens"] = num_tokens
        
        logging.info(f"Updated cell {cell_level}_{cell_index} with model {model_name} for operation {operation_name}")
    def save_model(self, model, operation_name: str, model_name: str, 
                training_trajectories_path: str) -> str:
        """
        Save a trained model to the appropriate cell in the pyramid.
        
        Args:
            model: The trained model object to save
            operation_name (str): Name of the operation (e.g., 'classification')
            model_name (str): Name/identifier for this specific model
            training_trajectories_path (str): Path to CSV file with training trajectories
            
        Returns:
            str: Path where the model was saved
        """
        # 1. Load training trajectories
        trajectories = self.load_trajectories(training_trajectories_path)
        if not trajectories:
            raise ValueError("No training trajectories found.")
        
        # 2. Find the enclosing cell
        cell = self._find_enclosing_cell_of_trajectory_list(trajectories)
        if not cell:
            raise ValueError("No enclosing cell found for training trajectories.")
        
        # 3. Get cell path and create operation subdirectory
        cell_level = cell["height"]
        cell_index = cell["index"]
        
        # Create path: modelsRepository/operation_name/level_index/
        operation_cell_path = os.path.join(
            self.models_repo_path, 
            operation_name, 
            f"{cell_level}_{cell_index}"
        )
        os.makedirs(operation_cell_path, exist_ok=True)
        
        # 4. Save the model
        model_filename = f"{model_name}.pt"  # or .pt for PyTorch
        model_path = os.path.join(operation_cell_path, model_filename)
        
        # Handle different model types
        if hasattr(model, 'save'):
            model.save(model_path)
        elif hasattr(model, 'save_model'):
            model.save_model(model_path)
        elif hasattr(model, 'state_dict'):  # PyTorch model
            import torch
            torch.save(model.state_dict(), model_path)
        else:
            # Fallback: use pickle
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # 5. Update pyramid metadata
        self._update_cell_with_model(operation_name, model_name, model_path, cell, len(trajectories))
        
        # 6. Save the updated pyramid
        self.save_pyramid()
        
        logging.info(f"Model saved to: {model_path}")
        return model_path    