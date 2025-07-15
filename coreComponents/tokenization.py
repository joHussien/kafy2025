# Tokenization.py

import h3
from shapely.geometry import Point
from typing import List, Dict

class Tokenization:
    @staticmethod
    def tokenize(points: List[Point], resolution: int) -> List[str]:
        """
        Converts a list of shapely.geometry.Point objects into H3 tokens at a specified resolution.

        Args:
            points (List[Point]): List of shapely.geometry.Point objects.
            resolution (int): H3 resolution.

        Returns:
            List[str]: H3 tokens corresponding to the input points.
        """
        return [h3.latlng_to_cell(point.y, point.x, resolution) for point in points]
    @staticmethod
    def is_spatial(result):
        """
        Checks whether the output of a model is a list of spatial tokens (e.g., H3).
        Returns True if the result is likely to be spatial and should be detokenized.
        """
        if not isinstance(result, list) or len(result) == 0:
            return False

        # Case 1: result is a list of tokens, and each is a valid H3 index
        if all(isinstance(token, str) and h3.h3_is_valid(token) for token in result):
            return True

        # (Optional) Add other token types in future
        # e.g., check if tokens follow a grid ID pattern or are in a known registry

        return False

    @staticmethod
    def tokenize_dataset(dataset: List[List[Point]], resolution: int) -> List[List[str]]:
        """
        Converts a list of trajectories to H3 tokens.

        Args:
            dataset (List[List[Point]]): List of trajectories, each a list of Points.
            resolution (int): H3 resolution.

        Returns:
            List[List[str]]: Tokenized trajectories.
        """
        return [Tokenization.points2tokens(traj, resolution) for traj in dataset]

    @staticmethod
    def from_geojson(geojson: Dict, resolution: int) -> List[List[str]]:
        """
        Converts a GeoJSON LineString dataset to tokenized form.

        Args:
            geojson (Dict): GeoJSON-like dictionary of LineStrings.
            resolution (int): H3 resolution.

        Returns:
            List[List[str]]: Tokenized trajectories.
        """
        tokenized = []
        for feature in geojson["features"]:
            coords = feature["geometry"]["coordinates"]
            points = [Point(lon, lat) for lon, lat in coords]
            tokenized.append(Tokenization.points2tokens(points, resolution))
        return tokenized