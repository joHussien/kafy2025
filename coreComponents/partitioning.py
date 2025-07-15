"""
Partitioning.py
"""

import os
import json
import logging
from math import sqrt
from typing import List, Tuple
from shapely.geometry import Point

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

        

        return operation_models, trajectories
    
    @staticmethod
    def load_trajectories(path: str) -> List[List[Point]]:
        """
        Loads a GeoJSON file containing LineString trajectories.
        
        Args:
            path (str): Path to the GeoJSON file.

        Returns:
            List[List[Point]]: A list of trajectories, where each trajectory is a list of shapely Point objects.
        """
        with open(path, 'r') as f:
            geojson_data = json.load(f)

        trajectories = []
        for feature in geojson_data.get("features", []):
            if feature["geometry"]["type"] == "LineString":
                coords = feature["geometry"]["coordinates"]
                trajectory = [Point(lon, lat) for lon, lat in coords]
                trajectories.append(trajectory)
            else:
                raise ValueError("Only LineString geometries are supported in trajectory input.")
        
        return trajectories
   
    def load_config(self):
        default_configs = {"H": 5, "L": 3, "build_pyramid_from_scratch": True}
        if not os.path.isfile(self.config_file):
            with open(self.config_file, "w", encoding="utf-8") as file:
                json.dump(default_configs, file, indent=4)
            raise Warning("Pyramid Configurations File not found. Assigned default configs.")

        with open(self.config_file, "r", encoding="utf-8") as file:
            config = json.load(file)
            self.pyramid_height = config.get("H", 5)
            self.pyramid_levels = config.get("L", 3)
            self.build_pyramid_flag = config.get("build_pyramid_from_scratch")

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

    def _update_cell_with_model(self, operation, cell, num_tokens):
        l = cell["height"]
        index = cell["index"]
        cell_path = os.path.join(self.models_repo_path, operation, f"{l}_{index}")
        os.makedirs(cell_path, exist_ok=True)
        cell.update({
            "model_path": cell_path,
            "occupied": True,
            "num_tokens": num_tokens
        })